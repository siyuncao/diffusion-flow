# diffusion-flow

A learning project. 
1. Flow matching on MNIST
2. Flow matching on constrained vectors
3. Diffusion on the MNIST (benchmark against 1.)

## Files

```
diffusion-flow/
├── README.md
├── notes.md              # derivations, diagnoses, comparisons
├── plan_flow.md          # what flow matching set out to do
├── plan_diffusion.md     # what diffusion set out to do
├── flow_mnist.py         # flow matching, UNet + Euler sampler
├── flow_simplex.py       # flow matching, MLP + simplex projection
├── diffusion_mnist.py    # DDPM, same UNet as flow_mnist.py
└── benchmark_mnist.py          # benchmark across VAE, GAN, flow matching, diffusion
```

## The architecture ladder

| architecture | the best loss | what the samples showed | diagnosis |
|---|---|---|---|
| MLP | 0.892 | digit-ish shapes buried in per-pixel static | no spatial prior — all 784 pixels treated as unordered, so it can't learn strokes |
| flat CNN | 0.245 | clean strokes, no global coherence | four 3×3 layers = 9×9 receptive field; each output pixel can't see the rest of the image |
| UNet | 0.183 | recognisable digits | downsampling gives deep layers a global view, skips carry the detail back |

Caveat: the CNN ran 10 epochs, the MLP and UNet 20. Not a controlled comparison, so the ranking is clear but the numbers aren't directly comparable.

## Flow matching vs diffusion

Same UNet, same data, same 20 epochs. Only the recipe changes.

| | flow matching | diffusion |
|---|---|---|
| path | straight line, `(1-t)·noise + t·data` | 1000 compounded tiny noise additions |
| `t` | float in [0,1] | integer index 0..999 |
| target | velocity `data - noise` | the noise itself |
| sampler | Euler, deterministic | DDPM, stochastic |
| steps at sampling | 100 | 1000 |
| sampling time, 16 images | 0.26s | 2.35s |
| sample quality | recognisable digits | recognisable digits, strokes slightly more solid (n=16 — an impression, not a result) |

Why diffusion needs 10× the steps: its path is *built* from 1000 tiny
additions, so undoing it wants 1000 tiny steps. Flow matching's path is a
straight line, and straight lines tolerate big steps.

## The simplex result

`flow_simplex.py` generates 12-dim vectors that must be non-negative and sum
to 1. Nothing in an MSE loss encodes that constraint, so raw samples fail.

The fix is a two-line projection — clamp negatives, renormalise. The question
is *where* it goes:

| projection | valid rows | generated std | training std |
|---|---|---|---|
| none | 62% | — | 0.048 |
| every Euler step | 100% | 0.004 | 0.048 |
| once after the loop | 100% | 0.045 | 0.048 |

Projecting every step compounds. Each renormalise pulls samples toward the
centre of the simplex, and the next step starts from the pulled position, so
100 small corrections stack into a collapse — 12× too narrow. Projecting once
gives the same guarantee and keeps the learned spread.

**The means matched in all three cases.** A check that only compared means
would have passed the collapsed run.

## All four (VAE, GAN, flow matching, diffusion), one matched setup

`benchmark_mnist.py` - written by Claude Code. 

| model | params | steps | sec/1000 | confidence | entropy |
|---|---|---|---|---|---|
| real MNIST (ref) | — | — | — | 0.9777 | 2.2938 |
| VAE (prior sample) | 320,804 | 1 | <0.01 | 0.7263 | 2.1813 |
| GAN | 114,064 | 1 | <0.01 | 0.7732 | 2.0913 |
| flow (Euler 100) | 1,882,561 | 100 | 15.61 | 0.8815 | 2.2498 |
| diffusion (DDPM 1000) | 1,882,561 | 1000 | 157.42 | 0.8666 | 2.2442 |
| diffusion (DDIM 50) | 1,882,561 | 50 | 7.78 | 0.8542 | 2.2375 |
