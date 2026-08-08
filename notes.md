# Note

## Phase 0

Cr, Mn, Fe, Co, Ni — the Cantor alloy, the most-studied HEA system (High-Entropy Alloy — a mix of 5+ metals in roughly equal amounts)

Al, Ti, V — lighter, add strength

Nb, Mo, Zr — refractory, high-temp stability

Cu — ductility

---
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


