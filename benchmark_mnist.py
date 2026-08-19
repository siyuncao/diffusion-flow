%%writefile benchmark_mnist.py
"""benchmark_mnist.py — one fair comparison of VAE, GAN, flow matching, diffusion.

Here every model starts from noise:

    VAE        z ~ N(0, I) -> decoder
    GAN        z ~ N(0, I) -> generator
    flow       x ~ N(0, I) -> Euler ODE integration
    diffusion  x ~ N(0, I) -> DDPM ancestral, or DDIM

What is held fixed across all four:
  * the data pipeline   — one transform, one DataLoader factory, same seed, so
                          every model sees the same batches in the same order
  * epochs
  * batch size
  * the sampling task   — --samples images from noise, timed the same way
  * the judge           — one small CNN classifier, trained once, scores everyone

What is deliberately *not* held fixed: architecture, parameter count, and the
optimizer settings each model needs to train at all (a GAN at lr=1e-3 collapses;
forcing one lr on all four would be a different kind of unfair). Parameter counts
are printed for every model so the capacity mismatch is visible rather than
hidden — the UNet is roughly 1.9M parameters against the VAE's 0.32M and the
GAN generator's 0.11M, so the flow/diffusion half of the table is carrying ~6x
and ~16x the capacity. That is part of the reading, not a defect to be argued
away: the exact counts are in the table, so nobody has to take the ranking at
face value.

Two quality numbers, both from the classifier:

  mean max-softmax   averaged over the samples. "Does this look confidently like
                     *some* digit?" High = clean digits. Blurry or garbled
                     samples land between classes and the confidence drops.

  class entropy      entropy of the histogram of predicted classes, in nats.
                     ln(10) = 2.303 is a perfectly uniform spread over the ten
                     digits. This is the mode-collapse detector: a GAN that has
                     learned to emit three beautiful 8s and nothing else scores a
                     *high* max-softmax and a low entropy. Neither number alone
                     catches it; together they do.

A row of real MNIST test images goes through the same two metrics as a
reference ceiling.

DDIM is here to answer a specific question: DDPM is stochastic and wants 1000
steps, flow matching is a deterministic ODE and is happy with 100. Drop the
re-noising term from DDPM and you have DDIM — a deterministic ODE over the same
trained network. At matched step counts it should behave like the flow sampler,
which is the sense in which the two methods are "the same thing". The script
asserts DDIM is bit-identical across two runs from the same x_T (and shows DDPM
is not, so the assertion is not vacuous).

Usage:
    python benchmark_mnist.py                          # defaults, needs a GPU to be quick
    python benchmark_mnist.py --epochs 20 --samples 1000
    python benchmark_mnist.py --sweep                   # step-count sweep, flow vs DDIM
    python benchmark_mnist.py --quick                   # tiny run, for checking the plumbing
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# 0. shared setup
# ---------------------------------------------------------------------------

# The canonical pipeline. [0,1] from ToTensor, then [-1,1] — the range the GAN's
# tanh, the flow model and the diffusion model all live in. The VAE is the one
# exception and it converts internally; see VAEModel.
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

T_DIFFUSION = 1000          # DDPM training/sampling horizon
GAN_LATENT = 100
VAE_LATENT = 10


def count_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


def get_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device):
    """cuda kernels are async — without this every timing below is a lie."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_loader(args, seed):
    """Same data, same batch size, same shuffle order for every model.

    The DataLoader gets its own seeded generator rather than relying on the
    global RNG, so model #4 sees exactly the batch sequence model #1 saw even
    though three trainings have burned global randomness in between.
    """
    train = datasets.MNIST(root=args.data_root, train=True, download=True, transform=TRANSFORM)
    if args.train_subset:
        train = torch.utils.data.Subset(train, range(args.train_subset))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(train, batch_size=args.batch_size, shuffle=True,
                      generator=g, drop_last=False)


# ---------------------------------------------------------------------------
# 1. the judge: a small MNIST classifier
# ---------------------------------------------------------------------------

class Classifier(nn.Module):
    """Small CNN. Not a research-grade evaluator — it just needs to be a stable,
    honest scorer of 'is this a digit, and which one'. Test accuracy is printed
    so you can see how much to trust it."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(args, device):
    torch.manual_seed(args.seed)
    loader = make_loader(args, seed=args.seed)
    test = datasets.MNIST(root=args.data_root, train=False, download=True, transform=TRANSFORM)
    test_loader = DataLoader(test, batch_size=512, shuffle=False)

    clf = Classifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for epoch in range(args.classifier_epochs):
        clf.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(clf(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  classifier epoch {epoch}: loss {total / len(loader):.4f}")

    clf.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (clf(x).argmax(1) == y).sum().item()
            n += y.numel()
    acc = correct / n
    print(f"  classifier test accuracy: {acc:.4f}")
    return clf, acc


@torch.no_grad()
def score(clf, samples, device):
    """samples: (N,1,28,28) in [-1,1]. Returns (mean max-softmax, class entropy in nats)."""
    clf.eval()
    confidences = []
    counts = torch.zeros(10, device=device)
    for i in range(0, samples.shape[0], 500):
        batch = samples[i:i + 500].to(device)
        probs = F.softmax(clf(batch), dim=1)
        conf, pred = probs.max(dim=1)
        confidences.append(conf)
        counts += torch.bincount(pred, minlength=10).float()
    conf = torch.cat(confidences).mean().item()
    p = counts / counts.sum()
    p = p[p > 0]                      # 0 * log 0 = 0, and log 0 is -inf, so drop empty classes
    entropy = max(-(p * p.log()).sum().item(), 0.0)   # max() only to kill -0.0 in the printout
    return conf, entropy, counts.cpu()


# ---------------------------------------------------------------------------
# 2. the four models
# ---------------------------------------------------------------------------
# Each wrapper exposes the same two methods:
#     train(args, device)      -> trains on the shared loader
#     sample(n, device, ...)   -> (n,1,28,28) in [-1,1], starting from noise
# and the same two attributes: .name and .modules (for the parameter count).

class VAEModel:
    """Encoder -> mu, logvar -> reparameterised z -> decoder. From vae.py.

    Sampling is the whole point of this file: z ~ N(0, I), decode, done. No
    encoder involved, no real digit involved. That is the fix to the unfair
    comparison — the VAE now plays the same game as the other three.

    The one pipeline concession: the decoder is sigmoid + BCE, which needs
    targets in [0,1], so it rescales the shared [-1,1] batch on the way in and
    rescales its samples back on the way out. Same bytes, same batches, same
    order — only an affine map inside the model.
    """

    name = "VAE"
    steps = 1

    def __init__(self, device):
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 200),
            nn.ReLU(),
        ).to(device)
        self.fc_mu = nn.Linear(200, VAE_LATENT).to(device)
        self.fc_logvar = nn.Linear(200, VAE_LATENT).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(VAE_LATENT, 200), nn.ReLU(),
            nn.Linear(200, 28 * 28), nn.Sigmoid(),
        ).to(device)
        self.modules = [self.encoder, self.fc_mu, self.fc_logvar, self.decoder]

    def train(self, args, device):
        params = [p for m in self.modules for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=1e-3)
        loader = make_loader(args, seed=args.seed)
        for epoch in range(args.epochs):
            rec_total = kl_total = 0.0
            n = 0
            for x, _ in loader:
                x = x.to(device).view(x.size(0), -1)
                x01 = x * 0.5 + 0.5                      # [-1,1] -> [0,1] for BCE
                opt.zero_grad()
                h = self.encoder(x01)
                mu, logvar = self.fc_mu(h), self.fc_logvar(h)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                rec = F.binary_cross_entropy(self.decoder(z), x01, reduction="sum")
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                (rec + kl).backward()
                opt.step()
                rec_total += rec.item()
                kl_total += kl.item()
                n += x.size(0)
            print(f"  VAE epoch {epoch}: rec {rec_total / n:.4f}  kl {kl_total / n:.4f}")

    @torch.no_grad()
    def sample(self, n, device, batch=None):
        self.decoder.eval()
        batch = batch or n
        out = []
        for i in range(0, n, batch):
            b = min(batch, n - i)
            z = torch.randn(b, VAE_LATENT, device=device)     # the prior, not an encoder
            x01 = self.decoder(z).view(b, 1, 28, 28)
            out.append(x01 * 2 - 1)                           # back to the shared [-1,1]
        return torch.cat(out)


class GANModel:
    """Generator + discriminator MLPs, alternating updates. From vae_gan_mnist/gan.py."""

    name = "GAN"
    steps = 1

    def __init__(self, device):
        self.generator = nn.Sequential(
            nn.Linear(GAN_LATENT, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 28 * 28), nn.Tanh(),
        ).to(device)
        self.discriminator = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 1), nn.Sigmoid(),
        ).to(device)
        # The discriminator is training scaffolding, not part of the generator
        # that gets timed at sampling — counted separately in the report.
        self.modules = [self.generator]
        self.aux_modules = [self.discriminator]

    def train(self, args, device):
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        loader = make_loader(args, seed=args.seed)
        for epoch in range(args.epochs):
            d_last = g_last = float("nan")
            for x, _ in loader:
                x = x.to(device).view(x.size(0), -1)
                bs = x.size(0)
                noise = torch.randn(bs, GAN_LATENT, device=device)
                fakes = self.generator(noise)
                real_labels = torch.ones(bs, 1, device=device)
                fake_labels = torch.zeros(bs, 1, device=device)

                d_opt.zero_grad()
                d_loss = (criterion(self.discriminator(x), real_labels)
                          + criterion(self.discriminator(fakes.detach()), fake_labels))
                d_loss.backward()
                d_opt.step()

                g_opt.zero_grad()
                g_loss = criterion(self.discriminator(fakes), real_labels)
                g_loss.backward()
                g_opt.step()

                d_last, g_last = d_loss.item(), g_loss.item()
            print(f"  GAN epoch {epoch}: D {d_last:.4f}  G {g_last:.4f}")

    @torch.no_grad()
    def sample(self, n, device, batch=None):
        self.generator.eval()
        batch = batch or n
        out = []
        for i in range(0, n, batch):
            b = min(batch, n - i)
            z = torch.randn(b, GAN_LATENT, device=device)
            out.append(self.generator(z).view(b, 1, 28, 28))   # already tanh -> [-1,1]
        return torch.cat(out)


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(),
    )


class UNet(nn.Module):
    """The same UNet flow_mnist.py and diffusion_mnist.py both use. Identical
    architecture for both, so the flow-vs-diffusion half of the table isolates
    the recipe (path, target, sampler) and nothing else."""

    def __init__(self):
        super().__init__()
        self.down1 = block(2, 64)        # 28x28, second channel is t
        self.down2 = block(64, 128)      # 14x14
        self.mid = block(128, 256)       # 7x7 — sees the whole image
        self.up2 = block(256 + 128, 128)
        self.up1 = block(128 + 64, 64)
        self.out = nn.Conv2d(64, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        t = t.expand(-1, 1, 28, 28)
        d1 = self.down1(torch.cat([x, t], dim=1))
        d2 = self.down2(self.pool(d1))
        m = self.mid(self.pool(d2))
        u2 = self.up2(torch.cat([self.up(m), d2], dim=1))
        u1 = self.up1(torch.cat([self.up(u2), d1], dim=1))
        return self.out(u1)


class FlowModel:
    """Flow matching: straight-line path, velocity target, Euler sampler.

    x_t = (1-t)*noise + t*x, and the network predicts x - noise. From flow_mnist.py.
    """

    name = "flow matching"

    def __init__(self, device, steps=100):
        self.net = UNet().to(device)
        self.modules = [self.net]
        self.steps = steps

    def train(self, args, device):
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loader = make_loader(args, seed=args.seed)
        for epoch in range(args.epochs):
            total = 0.0
            for x, _ in loader:
                x = x.to(device)
                noise = torch.randn_like(x)
                t = torch.rand(x.shape[0], 1, 1, 1, device=device)
                x_t = (1 - t) * noise + t * x
                opt.zero_grad()
                loss = F.mse_loss(self.net(x_t, t), x - noise)
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"  flow epoch {epoch}: loss {total / len(loader):.4f}")

    @torch.no_grad()
    def sample(self, n, device, batch=None, steps=None, x_init=None):
        self.net.eval()
        steps = steps or self.steps
        batch = batch or n
        dt = 1.0 / steps
        out = []
        for i in range(0, n, batch):
            b = min(batch, n - i)
            x = x_init[i:i + b].clone() if x_init is not None else torch.randn(b, 1, 28, 28, device=device)
            for s in range(steps):
                t = torch.full((b, 1, 1, 1), s * dt, device=device)
                x = x + self.net(x, t) * dt
            out.append(x)
        return torch.cat(out)


class DiffusionModel:
    """DDPM: 1000-step noise schedule, epsilon-prediction. From diffusion_mnist.py.

    Two samplers over the *same* trained weights:

      ddpm  — ancestral, the repo's sampler: walk all T steps, and at every step
              add fresh noise back in (`+ beta.sqrt() * randn`). Stochastic.

      ddim  — the same walk with the re-noising deleted and the trajectory
              subsampled to `ddim_steps` timesteps. Deterministic given x_T,
              which makes it an ODE solver over the diffusion path — structurally
              the same object as the flow sampler, just a curved path instead of
              a straight one.
    """

    name = "diffusion"

    def __init__(self, device, ddim_steps=50, T=T_DIFFUSION):
        self.net = UNet().to(device)
        self.modules = [self.net]
        self.T = T
        self.steps = self.T
        # DDIM subsamples the training schedule, so it can never use more steps
        # than the schedule has.
        self.ddim_steps = min(ddim_steps, T)
        self.betas = torch.linspace(1e-4, 0.02, self.T, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def train(self, args, device):
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loader = make_loader(args, seed=args.seed)
        for epoch in range(args.epochs):
            total = 0.0
            for x, _ in loader:
                x = x.to(device)
                noise = torch.randn_like(x)
                t = torch.randint(0, self.T, (x.shape[0],), device=device)
                ab = self.alpha_bar[t].view(-1, 1, 1, 1)
                x_t = ab.sqrt() * x + (1 - ab).sqrt() * noise
                opt.zero_grad()
                loss = F.mse_loss(self.net(x_t, (t / self.T).view(-1, 1, 1, 1)), noise)
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"  diffusion epoch {epoch}: loss {total / len(loader):.4f}")

    @torch.no_grad()
    def sample(self, n, device, batch=None, x_init=None):
        """DDPM ancestral sampler — all T steps, re-noising at every step."""
        self.net.eval()
        batch = batch or n
        out = []
        for i in range(0, n, batch):
            b = min(batch, n - i)
            x = x_init[i:i + b].clone() if x_init is not None else torch.randn(b, 1, 28, 28, device=device)
            for s in reversed(range(self.T)):
                t = torch.full((b, 1, 1, 1), s / self.T, device=device)
                eps = self.net(x, t)
                a, ab = self.alphas[s], self.alpha_bar[s]
                x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
                if s > 0:
                    x = x + self.betas[s].sqrt() * torch.randn_like(x)   # the re-noising
            out.append(x)
        return torch.cat(out)

    @torch.no_grad()
    def sample_ddim(self, n, device, batch=None, steps=None, x_init=None):
        """DDIM — DDPM minus the re-noising, on a subsampled timestep schedule.

        Per step: estimate x0 from the predicted noise, then re-noise that
        estimate to the *next* schedule timestep instead of to t-1. With no
        stochastic term this is a deterministic map from x_T to x_0.
        """
        self.net.eval()
        steps = steps or self.ddim_steps
        batch = batch or n
        # e.g. T=1000, steps=50 -> [0, 20, ..., 980], walked in reverse.
        ts = torch.linspace(0, self.T - 1, steps).long().tolist()
        out = []
        for i in range(0, n, batch):
            b = min(batch, n - i)
            x = x_init[i:i + b].clone() if x_init is not None else torch.randn(b, 1, 28, 28, device=device)
            for j in reversed(range(steps)):
                s = ts[j]
                ab = self.alpha_bar[s]
                # alpha_bar at "one schedule step earlier"; before the first
                # timestep there is no noise at all, so alpha_bar = 1.
                ab_prev = self.alpha_bar[ts[j - 1]] if j > 0 else torch.tensor(1.0, device=device)
                t = torch.full((b, 1, 1, 1), s / self.T, device=device)
                eps = self.net(x, t)
                x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps     # no randn term
            out.append(x)
        return torch.cat(out)


# ---------------------------------------------------------------------------
# 3. timing
# ---------------------------------------------------------------------------

def timed_sample(fn, n, device, warmup=None):
    """Wall-clock for n samples, cuda-synchronised on both sides.

    `warmup` is an optional zero-arg callable run first: the first cuda call of
    a process pays for context setup and per-shape kernel selection, which would
    otherwise be charged to whichever model happens to run first. It should
    touch the same tensor shapes as the real run but do far less work.

    The RNG state is saved and restored around the warm-up, so whether a model
    warms up or not does not change the noise it then samples from.
    """
    if warmup is not None:
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        warmup()
        sync(device)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    sync(device)
    t0 = time.perf_counter()
    samples = fn(n)
    sync(device)
    return samples, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 4. reporting
# ---------------------------------------------------------------------------

def print_table(rows, uniform_entropy, notes=()):
    head = (f"{'model':<22}{'params':>12}{'steps':>8}{'seconds':>10}{'ms/img':>9}"
            f"{'max-softmax':>13}{'entropy':>10}")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        params = f"{r['params']:,}" if r["params"] is not None else "-"
        steps = str(r["steps"]) if r["steps"] is not None else "-"
        secs = f"{r['seconds']:.2f}" if r["seconds"] is not None else "-"
        per = f"{r['seconds'] / r['n'] * 1000:.2f}" if r["seconds"] is not None else "-"
        print(f"{r['name']:<22}{params:>12}{steps:>8}{secs:>10}{per:>9}"
              f"{r['confidence']:>13.4f}{r['entropy']:>10.4f}")
    n = max(r["n"] for r in rows)
    print(f"\nseconds  wall-clock to generate {n} samples, cuda-synchronised on both sides")
    print(f"entropy  nats over the 10 predicted classes; a perfectly uniform spread is "
          f"ln(10) = {uniform_entropy:.4f}")
    for note in notes:
        print(note)


def save_grid(samples, path, n=16):
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("--save-grids needs matplotlib (pip install matplotlib); "
                         "everything else in this benchmark runs without it")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, n // 2, figsize=(n, 3))
    for img, ax in zip(samples[:n].cpu(), axes.flat):
        ax.imshow(img.view(28, 28).clamp(-1, 1) * 0.5 + 0.5, cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20, help="same for all four models")
    p.add_argument("--batch-size", type=int, default=128, help="same for all four models")
    p.add_argument("--samples", type=int, default=1000, help="samples per model for timing + scoring")
    p.add_argument("--sample-batch", type=int, default=250,
                   help="sampling minibatch; 1000 UNet images at once will OOM most GPUs")
    p.add_argument("--flow-steps", type=int, default=100, help="Euler steps for flow matching")
    p.add_argument("--ddim-steps", type=int, default=50,
                   help="DDIM steps (clamped to --diffusion-T)")
    p.add_argument("--diffusion-T", type=int, default=T_DIFFUSION,
                   help="DDPM schedule length: both the training horizon and the "
                        "number of ancestral sampling steps")
    p.add_argument("--classifier-epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--train-subset", type=int, default=0,
                   help="use only the first N training images (debugging)")
    p.add_argument("--sweep", action="store_true",
                   help="also sweep step counts for flow vs DDIM (the equivalence test)")
    p.add_argument("--sweep-steps", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    p.add_argument("--save-grids", default=None, metavar="DIR",
                   help="write a 16-image PNG grid per model into DIR")
    p.add_argument("--quick", action="store_true",
                   help="1 epoch, 200 samples, 2000 training images — plumbing check only")
    args = p.parse_args()
    if args.quick:
        args.epochs = 1
        args.classifier_epochs = 1
        args.samples = 200
        args.sample_batch = 100
        args.train_subset = 2000
        args.flow_steps = 10
        args.ddim_steps = 5
        args.diffusion_T = 20
        args.sweep_steps = [2, 5]
    return args


def main():
    args = parse_args()
    device = get_device(args.device)
    uniform_entropy = math.log(10)
    # cudnn's autotuner can pick different conv algorithms for the same shape on
    # different calls, which would show up as tiny numerical drift and break the
    # determinism assertion below for the wrong reason. It defaults to off; set
    # it explicitly so the assertion is testing the sampler, not the backend.
    torch.backends.cudnn.benchmark = False

    print("=" * 78)
    print("MNIST generative benchmark — VAE / GAN / flow matching / diffusion")
    print("=" * 78)
    print(f"device              {device}")
    print(f"epochs              {args.epochs}    (same for all four)")
    print(f"batch size          {args.batch_size}    (same for all four)")
    print(f"samples per model   {args.samples}")
    print(f"DDPM schedule T     {args.diffusion_T}")
    print(f"seed                {args.seed}")
    if device.type == "cpu":
        print(f"\nWARNING: on CPU the DDPM sampler is {args.diffusion_T} UNet forward passes "
              f"per batch."
              "\n         Expect hours. Use --quick to check the plumbing, or a GPU for real numbers.")
    if args.samples % args.sample_batch:
        print(f"note: {args.samples} samples is not a multiple of --sample-batch "
              f"{args.sample_batch}; the last batch is smaller.")

    # Warm-up batch shape for the UNet samplers: same shape as the timed run.
    warm_n = min(args.sample_batch, args.samples)

    print("\n[1/4] training the judge (small CNN classifier)")
    clf, clf_acc = train_classifier(args, device)

    rows = []
    grids = {}

    # Reference ceiling: real test digits, same metrics, same classifier.
    test = datasets.MNIST(root=args.data_root, train=False, download=True, transform=TRANSFORM)
    real = torch.stack([test[i][0] for i in range(min(args.samples, len(test)))])
    conf, ent, _ = score(clf, real, device)
    rows.append(dict(name="real MNIST (ref)", params=None, steps=None, seconds=None,
                     n=real.shape[0], confidence=conf, entropy=ent))

    print("\n[2/4] training the four models on identical batches")

    # ---- VAE -------------------------------------------------------------
    torch.manual_seed(args.seed)
    vae = VAEModel(device)
    print(f"  VAE params: {count_params(*vae.modules):,}")
    vae.train(args, device)
    torch.manual_seed(args.seed + 1)
    samples, secs = timed_sample(lambda n: vae.sample(n, device, batch=args.sample_batch),
                                 args.samples, device,
                                 warmup=lambda: vae.sample(min(16, args.samples), device))
    conf, ent, counts = score(clf, samples, device)
    rows.append(dict(name="VAE (prior sample)", params=count_params(*vae.modules), steps=vae.steps,
                     seconds=secs, n=args.samples, confidence=conf, entropy=ent))
    grids["vae"] = samples
    print(f"  VAE class histogram: {counts.int().tolist()}")

    # ---- GAN -------------------------------------------------------------
    torch.manual_seed(args.seed)
    gan = GANModel(device)
    print(f"  GAN params: {count_params(*gan.modules):,} generator "
          f"+ {count_params(*gan.aux_modules):,} discriminator (training only)")
    gan.train(args, device)
    torch.manual_seed(args.seed + 1)
    samples, secs = timed_sample(lambda n: gan.sample(n, device, batch=args.sample_batch),
                                 args.samples, device,
                                 warmup=lambda: gan.sample(min(16, args.samples), device))
    conf, ent, counts = score(clf, samples, device)
    rows.append(dict(name="GAN", params=count_params(*gan.modules), steps=gan.steps,
                     seconds=secs, n=args.samples, confidence=conf, entropy=ent))
    grids["gan"] = samples
    print(f"  GAN class histogram: {counts.int().tolist()}")

    # ---- flow matching ---------------------------------------------------
    torch.manual_seed(args.seed)
    flow = FlowModel(device, steps=args.flow_steps)
    print(f"  flow params: {count_params(*flow.modules):,}")
    flow.train(args, device)
    torch.manual_seed(args.seed + 1)
    samples, secs = timed_sample(
        lambda n: flow.sample(n, device, batch=args.sample_batch), args.samples, device,
        # 2 steps at the real sampling batch shape: warms the exact conv kernels,
        # costs ~1% of the timed run.
        warmup=lambda: flow.sample(warm_n, device, batch=warm_n, steps=2))
    conf, ent, counts = score(clf, samples, device)
    rows.append(dict(name=f"flow (Euler {args.flow_steps})", params=count_params(*flow.modules),
                     steps=args.flow_steps, seconds=secs, n=args.samples,
                     confidence=conf, entropy=ent))
    grids["flow"] = samples
    print(f"  flow class histogram: {counts.int().tolist()}")

    # ---- diffusion -------------------------------------------------------
    torch.manual_seed(args.seed)
    diff = DiffusionModel(device, ddim_steps=args.ddim_steps, T=args.diffusion_T)
    print(f"  diffusion params: {count_params(*diff.modules):,} "
          f"(identical UNet to flow matching — same architecture, different recipe)")
    diff.train(args, device)

    torch.manual_seed(args.seed + 1)
    samples, secs = timed_sample(
        lambda n: diff.sample(n, device, batch=args.sample_batch), args.samples, device,
        warmup=lambda: diff.sample_ddim(warm_n, device, batch=warm_n, steps=2))
    conf, ent, counts = score(clf, samples, device)
    rows.append(dict(name=f"diffusion (DDPM {diff.T})", params=count_params(*diff.modules),
                     steps=diff.T, seconds=secs, n=args.samples, confidence=conf, entropy=ent))
    grids["ddpm"] = samples
    print(f"  DDPM class histogram: {counts.int().tolist()}")

    # ---- DDIM: same weights, deterministic sampler -----------------------
    torch.manual_seed(args.seed + 1)
    samples, secs = timed_sample(
        lambda n: diff.sample_ddim(n, device, batch=args.sample_batch), args.samples, device,
        warmup=lambda: diff.sample_ddim(warm_n, device, batch=warm_n, steps=2))
    conf, ent, counts = score(clf, samples, device)
    rows.append(dict(name=f"diffusion (DDIM {diff.ddim_steps})", params=count_params(*diff.modules),
                     steps=diff.ddim_steps, seconds=secs, n=args.samples,
                     confidence=conf, entropy=ent))
    grids["ddim"] = samples
    print(f"  DDIM class histogram: {counts.int().tolist()}")

    print("\n[3/4] results")
    print(f"(classifier test accuracy {clf_acc:.4f} — the ceiling on how much these two "
          f"numbers can be trusted)")
    print_table(rows, uniform_entropy, notes=[
        f"GAN params are the generator only; the discriminator "
        f"({count_params(*gan.aux_modules):,}) exists at training time and is never sampled from.",
        "DDPM and DDIM are the same trained weights and the same parameter count — "
        "only the sampler differs.",
    ])

    print("\n[4/4] DDIM determinism check")
    check_determinism(diff, flow, args, device)

    if args.sweep:
        equivalence_sweep(diff, flow, clf, args, device)

    if args.save_grids:
        import os
        os.makedirs(args.save_grids, exist_ok=True)
        for key, s in grids.items():
            path = os.path.join(args.save_grids, f"{key}.png")
            save_grid(s, path)
            print(f"wrote {path}")


def check_determinism(diff, flow, args, device):
    """DDIM removes the only stochastic term in the sampler, so two runs from the
    same x_T must land on the same pixels. Assert it — and run DDPM from that
    same x_T to show the assertion has teeth (DDPM re-noises, so it must differ)."""
    n = min(args.sample_batch, args.samples, 64)   # a batch is plenty to prove a point
    torch.manual_seed(args.seed + 99)
    x_T = torch.randn(n, 1, 28, 28, device=device)

    a = diff.sample_ddim(n, device, batch=n, x_init=x_T)
    b = diff.sample_ddim(n, device, batch=n, x_init=x_T)
    ddim_diff = (a - b).abs().max().item()

    c = diff.sample(n, device, batch=n, x_init=x_T)
    d = diff.sample(n, device, batch=n, x_init=x_T)
    ddpm_diff = (c - d).abs().max().item()

    e = flow.sample(n, device, batch=n, x_init=x_T)
    f = flow.sample(n, device, batch=n, x_init=x_T)
    flow_diff = (e - f).abs().max().item()

    print(f"  DDIM,   two runs from the same x_T: max |diff| = {ddim_diff:.3e}   (expect 0)")
    print(f"  DDPM,   two runs from the same x_T: max |diff| = {ddpm_diff:.3e}   (expect > 0)")
    print(f"  flow,   two runs from the same x_T: max |diff| = {flow_diff:.3e}   (expect 0)")

    assert ddim_diff == 0.0, f"DDIM is not deterministic: max |diff| = {ddim_diff:.3e}"
    assert flow_diff == 0.0, f"flow Euler is not deterministic: max |diff| = {flow_diff:.3e}"
    assert ddpm_diff > 0.0, ("DDPM produced identical runs — the re-noising term is missing, "
                             "so the DDIM check above proves nothing")
    print("  OK: DDIM and flow Euler are deterministic maps x_T -> x_0; DDPM is not.")


def equivalence_sweep(diff, flow, clf, args, device):
    """When do diffusion and flow matching become the same thing?

    Both samplers are now deterministic ODE integrators over the same kind of
    path, differing in the path's shape: flow matching's is a straight line,
    diffusion's is the curved variance-preserving schedule. Run both at matched
    step counts and watch where they converge and where the straight line's
    tolerance for big steps shows up.
    """
    print("\n[+] step-count sweep: flow (Euler) vs diffusion (DDIM), same budget")
    head = f"{'steps':>7}{'flow sec':>11}{'flow conf':>11}{'flow ent':>10}" \
           f"{'ddim sec':>11}{'ddim conf':>11}{'ddim ent':>10}"
    print(head)
    print("-" * len(head))
    for steps in args.sweep_steps:
        torch.manual_seed(args.seed + 1)
        fs, f_secs = timed_sample(
            lambda n, s=steps: flow.sample(n, device, batch=args.sample_batch, steps=s),
            args.samples, device)
        f_conf, f_ent, _ = score(clf, fs, device)

        torch.manual_seed(args.seed + 1)
        ds, d_secs = timed_sample(
            lambda n, s=steps: diff.sample_ddim(n, device, batch=args.sample_batch, steps=s),
            args.samples, device)
        d_conf, d_ent, _ = score(clf, ds, device)

        print(f"{steps:>7}{f_secs:>11.2f}{f_conf:>11.4f}{f_ent:>10.4f}"
              f"{d_secs:>11.2f}{d_conf:>11.4f}{d_ent:>10.4f}")
    print(f"both columns are deterministic ODE solvers over {args.epochs} epochs of training;\n"
          f"the gap that remains at matched steps is the path shape, not the method.")


if __name__ == "__main__":
    main()
