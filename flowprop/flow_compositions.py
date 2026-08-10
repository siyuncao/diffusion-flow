import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load
a = torch.rand(10000, 12)
a = a / a.sum(dim=1, keepdim=True)

# 2. Model
model = nn.Sequential(
    nn.Linear(13, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 12),
).to(device)

# 3. Loss
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Train
a = a.to(device)
for epoch in range(2000):
    noise = torch.randn_like(a)
    t = torch.rand(a.shape[0], 1, device=device)
    x_t = (1-t)*noise + t*a
    optimizer.zero_grad()
    outputs = model(torch.cat([x_t, t], dim=1))
    loss = criterion(outputs, a - noise)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Euler sampler
x = torch.randn(1000, 12, device=device)
steps = 100
dt = 1.0 / steps
with torch.no_grad():
    for i in range(steps):
        t = torch.full((1000, 1), i * dt, device=device)
        x = x + model(torch.cat([x, t], dim=1)) * dt
        
x = x.clamp(min=0) # move out of the loop stops the correction from compounding.
x = x / x.sum(dim=1, keepdim=True)

# 6. Result
print("generated means:", x.mean(dim=0))
print("training means:  ", a.mean(dim=0))
print("generated std:", x.std(dim=0))
print("training std:  ", a.std(dim=0))
