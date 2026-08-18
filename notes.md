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

### MLP
<img width="429" height="110" alt="Screenshot 2026-08-10 at 16 04 17" src="https://github.com/user-attachments/assets/30adc572-0248-4891-a16f-9880498c37a3" />

### CNN
<img width="423" height="109" alt="Screenshot 2026-08-10 at 16 04 40" src="https://github.com/user-attachments/assets/c58328a6-8a21-4f88-ad58-c161a5095efc" />

### UNet
<img width="943" height="235" alt="Screenshot 2026-08-10 at 16 05 08" src="https://github.com/user-attachments/assets/5f79e034-6c59-41b6-a1c4-c68a8f64bd19" />

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
