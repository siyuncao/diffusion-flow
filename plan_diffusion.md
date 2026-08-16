# Diffusion on MNIST

**What this is:** implement DDPM on MNIST, then compare it against the flow
matching model already in this repo.

**Why:** same goal (noise → data), different mechanism, same dataset and same
UNet. That isolates what the mechanism actually changes.

**Tags:** 🧠 = derive/write by hand, must be able to justify. 🤖 = vibe code, already covered.

- Phase 0: Noise schedule — the piece flow matching doesn't have.
- Phase 1: DDPM on MNIST. Same UNet as `flow_mnist.py`, three changes.
- Phase 2: Compare against flow matching.

---

## Phase 0 — Noise schedule 🧠

```python
T = 1000
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
```

**Done when:** I can say what `alpha_bar` does from t=0 to t=T, and why
diffusion needs a schedule at all when flow matching just uses `t` directly.

---

## Phase 1 — DDPM on MNIST 🧠

Copy `flow_mnist.py` → `ddpm_mnist.py`. The loader, UNet, optimiser and
training-loop shape all stay. Three things change:

1. **`t` is now an integer index**, not a float in [0,1]. Sample
   `t = torch.randint(0, T, (batch,))` and look up `alpha_bar[t]`.
2. **Forward process:**
   `x_t = sqrt(alpha_bar[t])*x + sqrt(1-alpha_bar[t])*noise`
   — weighted, not linear interpolation.
3. **Target is `noise`**, not velocity `x - noise`.

Then replace the Euler sampler with the DDPM sampler: step backwards
t=T→0, adding fresh noise back at each step (stochastic, unlike Euler).

**Verification gate:**
- loss decreases
- samples are recognisable digits
- fixed seed → identical sample

---

## Phase 2 — Compare

Same UNet, same data, same epochs.

| | flow matching | diffusion |
|---|---|---|
| path | straight line | schedule-defined |
| `t` | float in [0,1] | integer index 0..T |
| target | velocity `x - noise` | `noise` |
| sampler | Euler, deterministic | DDPM, stochastic |
| steps at sampling | 100 | ? |
| sample quality | recognisable digits | ? |

**Done when:** the table is filled in, *and* I can answer: why does diffusion
need so many more sampling steps? what would make the two equivalent?
