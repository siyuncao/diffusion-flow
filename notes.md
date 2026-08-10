# Note


## Phase 1

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
| assumes | nothing about input order | neighbours are related | neighbours related *and* structure exists at multiple scales |
| shape | flat vector in, flat vector out | image in, image out | image in, image out, via a bottleneck |
| suits | tabular / unordered features | images, local patterns | images where output must match input pixel-for-pixel (generation, segmentation) |

---
## Phase 2
default things in every PyTorch training loop
```python
optimizer.zero_grad()     # clear last step's gradients
outputs = model(inputs)   # forward: predict
loss = criterion(outputs, targets)   # measure error
loss.backward()           # backward: compute gradients
optimizer.step()          # update the weights
```

**1st Result — verification gate FAILS.**

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
