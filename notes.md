# Note

## Flow matching - Phase 1
### Data loader: transform explained
```python
transform=transforms.Compose([ # Compose is added first in order to chain the latter (ToTensor, Normalize) together in sequence
transforms.ToTensor(), # The MNIST file on disk stores 784 bytes per digit.
                       # ToTensor() reads those bytes into a PyTorch tensor and divides by 255, mapping everything into [0, 1].
transforms.Normalize((0.5,), (0.5,)) # maps [0, 1] → [-1, 1]
```

why print out the average loss?

context: MNIST has 60,000 images. You set batch_size=32, so the loader hands the model 32 at a time.
60,000 ÷ 32 ≈ 1875. That's how many batches per epoch.

```python
for epoch in range(5): # go through the entire dataset for 5 times
    total = 0
    for x, _ in train_loader:
      ...
      total += loss.item()
      ...

    print(f"Epoch {epoch}, Loss: {total / len(train_loader):.4f}")
```

- Originally: the loss of the last batch — that batch's 32 images averaged, but only one batch out of ~1900.
- Now: all ~1900 batches averaged.

because batch is random (after shuffled), so the last batch has no correlation to the previous batch, rather, it's randomly picked so the number might be high or low. now averaging all makes sure the loss keeps going down rather than bounces around

### Models compared

|  | MLP | CNN | UNet |
|---|---|---|---|
| building block | `nn.Linear` | `nn.Conv2d` | `nn.Conv2d` + pooling + upsampling |
| connectivity | every input to every neuron | small sliding window | sliding window at several scales |
| weights | unique per position | shared across positions | shared, plus skip connections |
| **assumes** | **nothing → must learn "pixel 5 and 6 are adjacent" from data** | **neighbours are related → gets locality for free** | **structure at multiple scales → gets local and global for free** |
| shape | flat vector in, flat vector out | image in, image out | image in, image out, via a bottleneck |
| suits | tabular / unordered features | images, local patterns | images where output must match input pixel-for-pixel (generation, segmentation) |

**MLP**

<img width="100%" alt="MLP architecture" src="https://github.com/user-attachments/assets/30adc572-0248-4891-a16f-9880498c37a3" />

**CNN**

<img width="100%" alt="CNN architecture" src="https://github.com/user-attachments/assets/c58328a6-8a21-4f88-ad58-c161a5095efc" />

**UNet**

<img width="100%" alt="UNet architecture" src="https://github.com/user-attachments/assets/5f79e034-6c59-41b6-a1c4-c68a8f64bd19" />

### UNet notes
elements: down path, bottleneck, up path

- Resolution = height × width. 28×28, then 14×14, then 7×7. How many positions.
- Channels = how many numbers stored at each position. Starts at 2 (image + t), grows to 64, 128, 256. How many features per position.

(Pool changes resolution; helper changes channels)

### what's noise

noise is composed of random numbers with the same shape as x. 

- at training: `x_t = (1-t)*noise + t*data` blends the training data and noise. `t` represents how far along the path from noise to data, 0 for pure noise and 1 for pure data. And `t` is random for each batch
- at sampling: this provides somewhere to start, and the trained model steps it all the way to a digit. No learning happens here (weights are frozen). `t` walks 0 → 1 in 100 equal steps (`dt = 1/100`)

### why noise at all:

you want *new* digits. A model that only maps digits to digits could never produce one that doesn't already exist. **Randomness is the
source of novelty**, and it's why every sample comes out different (same seed →
same sample).

### classifier vs generative

|  | classifier | generative |
|---|---|---|
| input | data | randomness |
| output | a label | new data |
| evaluate with | accuracy on test data | do the samples look right |
| **examples** | **MLP** | **VAE, GAN, flow matching, diffusion** |

---
## Flow matching - Phase 2
default things in every PyTorch training loop
```python
optimizer.zero_grad()     # clear last step's gradients
outputs = model(inputs)   # forward: predict
loss = criterion(outputs, targets)   # measure error
loss.backward()           # backward: compute gradients
optimizer.step()          # update the weights
```

### 1st Result — verification gate FAILS

| check | value |
|---|---|
| min value | −0.168 |
| row sums (sample) | 0.86 – 1.06 |
| fraction of entries negative | 4.5% |
| **fraction of rows valid** | **62%** |

The model learned the shape of the answer but not the rule. It knows compositions look like twelve smallish numbers near 0.08 that roughly sum to 1 — that's why sums land in 0.86–1.06 and the worst negative is only −0.17. It has genuinely absorbed the distribution.

- Negative: an element with fraction −0.17 means "minus 17% chromium." There's no such thing. That's why the row is thrown out — not because it's a bad alloy, but because it isn't an alloy.
- load part for training, sampler part for generating new

when there are negatives appear:
- why not change the loss part: A loss penalty makes violations rarer, not impossible. You'd go 62% → maybe 90%, and your gate still says 100% error. Soft pressure can't produce a hard guarantee.
- instead, fix the sampler: whatever comes out of the Euler step, clip negatives to 0, divide by the row sum. Now the row is non-negative and sums to 1, by construction, every time.
```python
x = x.clamp(min=0)
x = x / x.sum(dim=1, keepdim=True)
```


**Simplex fix: project once, after the Euler loop.**

**2nd Result (final)**

| projection | valid rows | generated std | training std |
|---|---|---|---|
| none | 62% | — | 0.048 |
| every Euler step | 100% | 0.004 | 0.048 |
| once at the end | 100% | 0.045 | 0.048 |

---
## Diffusion - Phase 0 (noise)

### noise: flow matching vs diffusion

|  | flow matching | diffusion |
|---|---|---|
| how noise mixes in | straight line, `(1-t)·noise + t·data` | 1000 tiny additions |
| the weights | `t` and `1-t` | `sqrt(alpha_bar)` and `sqrt(1-alpha_bar)` |
| where the weights come from | you pick the formula: a straight line | compounding 1000 tiny steps |
| `t` | float, 0 to 1 | integer, 0 to 999 |

Same journey. One drawn as a line, one built from small steps.

`alpha_bar[t]` = how much of the original image is left after t noising steps. It's `cumprod(1 - betas)`, so it starts near 1 and decays toward 0 — slowly at first, then fast.

---
## Diffusion - Phase 1

### Samplers: Euler vs DDPM

|  | Euler (flow matching) | DDPM (diffusion) |
|---|---|---|
| direction | t: 0 → 1 | t: T → 0 |
| model predicts | velocity (which way to move, since training targeted data - noise) | noise (how much noise is in the image, so the sampler can remove it) |
| update | `x + v*dt` | subtract predicted noise, rescale |
| extra noise per step | none (the path is deterministic) | added back every step (it mirrors the forward process (data → noise), which added noise at every step, so the reverse (noise → data) must too)|
| deterministic (Same input always gives the same output)? | yes (because it adds nothing random) | no (injects randomness 1000 times) |
| steps | 100 (the path is straight, so big steps are fine) | 1000 (built as 1000 tiny noise-additions, so it undoes them in 1000 tiny steps)|

**Q: You can estimate x₀ in one line. Why take 1000 steps instead of returning it?**
You only have a local guess. At t=999 the estimate is a blurry average of all
digits. It only sharpens as `x_t` gets cleaner.

```python
x₀ = (x - (1 - ab).sqrt() * eps) / ab.sqrt()   # the estimate — don't trust it yet
```

**Q: So what do you do with the estimate?**
Take your estimated x₀, and run the forward process on it to noise level t−1.
(Keeping the noise already in `x_t`, adding only the difference.)

```python
x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
if i > 0:
    x = x + betas[i].sqrt() * torch.randn_like(x)      # the difference
```
### when to use `with`
```python
with open('file.txt') as f:     # closes the file afterwards
    data = f.read()

with torch.no_grad():           # turns gradient tracking back on afterwards
    ...
```

`with` = do this setup, run my block, then guarantee the cleanup happens.

---
## Diffusion - Phase 2

### compare flow matching to diffusion
| | flow matching | diffusion |
|---|---|---|
| steps at sampling | 100 | 1000 |
| sampling time, 16 images | 0.26s | 2.35s |

sampling time explain: diffusion is 9.0× slower for 10× the steps — near-linear, as the step count predicts.

### Fair benchmark of VAE, GAN, flow matching, diffusion

see code at `benchmark_mnist.py`

**[Overview of different types of generative models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)**

<img width="925" height="639" alt="Screenshot 2026-08-19 at 13 57 51" src="https://github.com/user-attachments/assets/9b60c7ba-db41-44df-8fd2-c552d6b0a2df" />

**judge**

A small CNN classifier trained on real MNIST only, then frozen.

```python
class Classifier(nn.Module):
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
```

**benchmark**

| model | params | steps | sec/1000 | confidence | entropy |
|---|---|---|---|---|---|
| real MNIST (ref) | — | — | — | 0.9777 | 2.2938 |
| VAE (prior sample) | 320,804 | 1 | <0.01 | 0.7263 | 2.1813 |
| GAN | 114,064 | 1 | <0.01 | 0.7732 | 2.0913 |
| flow (Euler 100) | 1,882,561 | 100 | 15.61 | 0.8815 | 2.2498 |
| diffusion (DDPM 1000) | 1,882,561 | 1000 | 157.42 | 0.8666 | 2.2442 |
| diffusion (DDIM 50) | 1,882,561 | 50 | 7.78 | 0.8542 | 2.2375 |

- confidence = mean max-softmax from the judge (98.5% test acc).
- entropy = spread over the 10 predicted classes; uniform is ln(10) = 2.3026.

**equivalence conditions for diffusion and flow matching** 

Delete the re-noising line from DDPM and it becomes a deterministic ODE solver — the same *kind* of object as
the Euler sampler. It then inherits flow matching's tolerance for big steps:
20× fewer steps, 20× faster, quality essentially unchanged (0.8542 vs 0.8666).
What still differs is the path shape — flow matching's is straight by
construction, diffusion's is curved by `alpha_bar`.

Determinism, exact:
- DDIM, two runs from same `x_T`: 0.000e+00
- flow, two runs from same `x_T`: 0.000e+00
- DDPM, two runs from same `x_T`: 2.128e+00

**Caveats.**
- The UNet models carry 1.88M params vs the VAE's 0.32M and GAN's 0.11M (6× and
  16×). "Flow matching wins on quality" is partly a capacity story.
- DDIM being faster than flow is 50 steps vs 100 on an identical network, not
  diffusion being faster. At matched steps they'd be close.
- VAE and GAN sample in one forward pass — <0.01s vs 15.61s. That speed gap is
  the real reason GANs still get used.
- Nothing reaches real MNIST (0.9777). The gap is 20 epochs on a free T4.

**When to use which model**

The quality column can't decide it here in my benchmark. My UNet models had 16× the GAN's
parameters, so method and capacity are confounded. Speed decides instead since the
gap is 1000× and doesn't depend on model size.

| | when to pick it |
|---|---|
| **GAN** | you need one-step generation. <0.01s vs 15.61s is a 1000× gap. Real-time, interactive, on-device. |
| **flow matching** | default for new work. Best quality here, fewer steps than DDPM, simplest training objective. |
| **diffusion** | when you want the ecosystem — pretrained weights, tooling, papers to borrow from. And use DDIM, not DDPM. |
| **VAE** | not as a generator. As a component — compress to a latent, then run diffusion there. That's Stable Diffusion. |

Flow matching and diffusion are the same family, so pick on ecosystem and step
count, not quality. GAN is a genuinely different trade: worse and less stable,
but a thousand times faster.

What this table can't settle is whether the quality gap holds at scale — my
comparison is confounded by parameter count (1.88M vs 0.11M). For that, the
literature: diffusion and flow matching win on quality at scale, comfortably.
