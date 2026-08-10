# Flow Matching on Two Data Types

**What this is:** learn flow matching from the math, then test the same
mechanism on two structurally different datasets — images (MNIST) and
simplex-constrained vectors — to see what has to change with the data and what
doesn't.

**Why this project:** flow matching is the next rung on the generative-model
path (MLP → VAE → GAN → flow). Running one mechanism on two data types
isolates what belongs to flow matching itself and what belongs to the domain.

**Tags:** 🧠 = derive/write by hand, must be able to justify. 🤖 = vibe code, already covered.

- Phase 0: Derive the loss by hand.
- Phase 1: Flow matching on MNIST. Train, sample with Euler. Failure is visible here.
- Phase 2: Same loss, same sampler, simplex-constrained vectors. Core problem: values must be non-negative and sum to 1 — pick a fix and justify it.
- Phase 3: Conditional generation. Stretch.

---
```
flow-proposer/
├── notes.md
├── README.md
├── .gitignore
├── flowprop/
│   ├── flow.py            # loss + Euler sampler (shared by both datasets)
│   ├── models.py          # MLP, CNN, UNet
│   └── simplex.py         # 🧠 the real design decision
├── experiments/
│   ├── 01_mnist.py
│   └── 02_compositions.py
└── tests/
    └── test_flow.py
```
---

## [Flow matching vs diffusion](https://harshm121.medium.com/flow-matching-vs-diffusion-79578a16c510)

- **Diffusion Models** gradually add noise to data until it becomes pure noise, then learn to reverse this process. Think of it as slowly dissolving a photograph in acid until it becomes a random blur, then learning how to reconstruct the photograph from the blur.

- **Flow Matching** creates a continuous path (or flow) between noise and data distributions. Think of it as defining a smooth transportation plan that morphs noise into structured data, similar to watching a time-lapse of clay being sculpted from a random blob into a detailed statue.

---

## Phase 0 — Derive the loss 🧠

Sample `t ~ U(0,1)`, interpolate `x_t = (1-t)·noise + t·data`, target velocity
`v = data - noise`, loss = MSE against the net's prediction. Four lines — but
be able to explain *why* the target is `data - noise`.

**Done when:** the four lines are written from scratch and the target is
justified as the derivative of the straight-line path.

---

## Phase 1 — Flow matching on MNIST 🧠

Learn the mechanism somewhere failure is visible.

1. Load MNIST, normalise to [-1, 1].
2. Write the model: takes `(x_t, t)`, returns a velocity.
3. Write the 4-line loss.
4. Train loop. Watch loss drop.
5. Write the Euler sampler 🤖: start from noise, step 100 times.
6. Plot samples.

**why use a neural network?** Since at sampling time we only have `x_t` and
don't know the velocity `x₁ - x₀`, we need a neural network to learn to
predict it.

**why use Euler?** Since we start from noise and only know the velocity at the
current point, we take one small step in that direction, ask the model again,
and repeat — thus 100 steps of `x = x + v*dt` gets us from noise to a digit.
That's Euler.

**Verification gate:**
- loss decreases
- samples at `t=1` are recognisable digits
- fixed seed → identical sample (reproducibility)

**Architecture ladder.** Start with an MLP to prove the loop runs, then only
change architecture when the samples show a specific failure — and record the
diagnosis, not just the swap. MLP → CNN → UNet, each step earned.

**Done when:** digits look right, *and* I can answer: why is the path straight?
what changes with 10 vs 100 sampling steps? why is there no ELBO here, unlike
the VAE?

---

## Phase 2 — Flow matching on simplex-constrained vectors 🧠

Same loss, same sampler, different data: 12-dim vectors that must be
non-negative and sum to 1.

- [ ] Training set: sample valid vectors directly — 12 uniform draws per row,
      divided by the row sum.
- [ ] **Architecture: back to an MLP.** 12 unordered numbers have no spatial
      structure, so convolution has nothing to exploit. Architecture follows
      the data's structure, not the task.
- [ ] **The core design problem: the simplex constraint.** Naive flow matching
      will happily emit `[-0.3, 0.8, 0.5]` — invalid. Pick one and justify it:
      1. generate in an unconstrained transformed space (softmax / logit) and map back
      2. project onto the simplex after each Euler step
      3. a simplex-aware flow
- [ ] Write the justification in `notes.md` — this is the Matérn-vs-RBF moment
      of this project.

**Verification gate:**
- 100% of generated samples are valid (non-negative, sum to 1)
- mean of generated ≈ mean of training
- **std of generated ≈ std of training** — means alone are not enough. A
  collapsed distribution can match on means and still be useless.

**Scope note:** synthetic vectors, no domain filtering. This is a mechanism
test of whether flow matching can learn a constrained distribution.

---

## Phase 3 — Conditional generation (stretch)

Guide generation toward a target instead of sampling the whole distribution.
This is the conditioning/guidance skill the rest of the roadmap needs.
