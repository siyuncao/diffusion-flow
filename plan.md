# Flow Matching → Generative Proposer

**What this is:** learn flow matching from the math, then point it at alloy compositions and
plug it into the closed discovery loop as a generative candidate proposer.

**Why this project:** it sits on both tracks at once — flow matching is the next rung on the
generative-model path (MLP → VAE → GAN → flow), and a generative proposer is the upgrade that
turns a fixed-pool BO loop into an actual discovery engine.

**Tags:** 🧠 = derive/write by hand, must be able to justify. 🤖 = vibe code, already covered.


---
```
flow-proposer/
├── notes.md
├── README.md
├── .gitignore
├── flowprop/
│   ├── flow.py            # loss + model + Euler sampler
│   ├── simplex.py         # 🧠 the real design decision
│   ├── compositions.py    # AlloySpace sampling
│   └── loop.py            # generate → featurize → GP+EI → refit
├── experiments/
│   ├── 01_mnist.py
│   ├── 02_compositions.py
│   └── 03_benchmark.py
└── tests/
    └── test_flow.py
```
---

## [Flow matching vs diffusion](https://harshm121.medium.com/flow-matching-vs-diffusion-79578a16c510)

- **Diffusion Models** gradually add noise to data until it becomes pure noise, then learn to reverse this process. Think of it as slowly dissolving a photograph in acid until it becomes a random blur, then learning how to reconstruct the photograph from the blur.

- **Flow Matching** creates a continuous path (or flow) between noise and data distributions. Think of it as defining a smooth transportation plan that morphs noise into structured data, similar to watching a time-lapse of clay being sculpted from a random blob into a detailed statue.

---

## Phase 0 — Choose the representation

Flow matching wants **continuous** space. Alloy composition vectors are the
right one: continuous, low-dimensional, physically meaningful, and already supported by
`AlloySpace`.

Open notes.md and list ~12 elements with one line of "why" each.

---

## Phase 1 — Flow matching on MNIST 🧠

1. Load MNIST, flatten to 784.
2. Write the MLP: input 785 (image + t), output 784.
3. Write the 4-line loss.
4. Train loop. Watch loss drop.
5. Write the Euler sampler: start from noise, step 100 times.
6. Plot samples.

why use MLP? Since at sampling time we only have x_t and don't know the velocity x₁ - x₀, we need a neural network to learn to predict it. Thus we used MLP in phase 1.

why use Euler? Since we start from noise and only know the velocity at the current point, we take one small step in that direction, ask the model again, and repeat — thus 100 steps of x = x + v*dt gets us from noise to a digit. That's Euler.

---

## Phase 2 — Flow matching on compositions 🧠

Same code, chemical data.

- [ ] Training set: sample valid compositions from `AlloySpace`.
- [ ] **The core design problem: the simplex constraint.** Compositions must be
      non-negative and sum to 1. Naive flow matching will happily emit
      `[-0.3, 0.8, 0.5]` — chemically meaningless. Pick one and justify it:
      1. generate in an unconstrained transformed space (softmax / logit) and map back
      2. project onto the simplex after each Euler step
      3. a simplex-aware flow
- [ ] Write the justification in `notes.md` — this is the Matérn-vs-RBF moment of this
      project.

**Verification gate:**
- 100% of generated samples are valid compositions (non-negative, sum to 1)
- element-frequency histogram of generated ≈ training distribution

---

## Phase 3 — Plug into the loop 🧠

Replace the fixed candidate pool with the generative proposer.

- [ ] Each round: generate N candidates → featurize (Magpie) → score with GP + EI →
      measure the best → refit.
- [ ] **Benchmark that makes it a result:** generative proposer vs enumerated pool, same
      budget, ~20 seeds, mean ± std.
- [ ] **Honest scope note (non-negotiable):** the oracle is still a synthetic benchmark.
      Generated candidates are scored by a model, not made in a lab.

**Expect the null.** The enumerated pool may already cover the good region, in which case
generating doesn't help — that is a real finding and it gets reported, exactly like the
BO-vs-random result in `discovery-loop`.

---

## Phase 4 — Conditional generation (stretch)

Guide generation toward high predicted hardness / low density instead of sampling the whole
distribution. This is what makes a proposer useful rather than decorative, and it's the
conditioning/guidance skill the rest of the roadmap needs.

**Only after Phases 1–3 are solid.**
