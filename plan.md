# Flow Matching → Generative Proposer

**What this is:** learn flow matching from the math, then point it at alloy compositions and
plug it into the closed discovery loop as a generative candidate proposer.

**Why this project:** it sits on both tracks at once — flow matching is the next rung on the
generative-model path (MLP → VAE → GAN → flow), and a generative proposer is the upgrade that
turns a fixed-pool BO loop into an actual discovery engine.

**Tags:** 🧠 = derive/write by hand, must be able to justify. 🤖 = vibe code, already covered.

---

## Flow matching vs diffusion (one paragraph, for the README later)

Both learn noise → data along a path. **Diffusion (DDPM)** is discrete-time and stochastic:
add Gaussian noise over ~1000 steps, train a net to predict the noise, walk backwards. Math
descends from the VAE's ELBO. **Flow matching** is continuous-time and deterministic: define
a straight path `x_t = (1-t)·noise + t·data`, train a net to predict the *velocity*, sample by
following the vector field (an ODE). No ELBO, no variational bound — just regression.

Diffusion asks *"what noise was added here?"*; flow matching asks *"which way do I move now?"*
Same destination, simpler bookkeeping. Flow matching is what modern large models (SD3, Flux)
use, because straight paths mean fewer sampling steps.

---

## Phase 0 — Choose the representation

Flow matching wants **continuous** space. Alloy composition vectors are the
right one: continuous, low-dimensional, physically meaningful, and already supported by
`AlloySpace`.

- [ ] Pick ~12 candidate elements. Write down **why** — the constraints that matter
      (cost, density, toxicity, availability) and the HEA formability rules
      (atomic size mismatch δ, VEC, mixing enthalpy).
- [ ] Reference: Wikipedia "High-entropy alloys" for the design rules; `elements.py`,
      `features.py` (`size_mismatch_delta`, `valence_electron_concentration`) and
      `space.py` for how they're already implemented here.
- [ ] Record the element set and the reasoning in `notes.md`. **This is the chemistry
      judgment the project rests on — it does not get outsourced.**

**Done when:** a justified element list exists, and I can defend each inclusion/exclusion.

---

## Phase 1 — Flow matching on MNIST 🧠

Learn the mechanism somewhere failure is visible.

- [ ] **Derive the loss by hand.** Sample `t ~ U(0,1)`, interpolate
      `x_t = (1-t)·noise + t·data`, target velocity `v = data - noise`, loss = MSE against
      the net's prediction. Four lines — but be able to explain *why* the target is
      `data - noise`.
- [ ] Model: small UNet (or MLP first, to prove the loop works).
- [ ] Sampler 🤖: Euler integration from noise, `x ← x + v·dt`.

**Verification gate:**
- loss decreases
- samples at `t=1` are recognisable digits
- fixed seed → identical sample (reproducibility)

**Done when:** digits look right, *and* I can answer: why is the path straight? what changes
with 10 vs 100 sampling steps? why is there no ELBO here, unlike the VAE?

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
