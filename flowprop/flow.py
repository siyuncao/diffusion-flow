import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load (copied from VAE)
train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))

test_data = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# 2. Conv model
# Why a class instead of nn.Sequential: Sequential takes one input, but we
# need two (x_t and t). A class lets forward() accept both.
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Conv2d looks at 3x3 patches, so neighbouring pixels are related
            # by construction — the MLP had to learn that from scratch and couldn't.
            # padding=1 keeps the image 28x28 throughout.
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(),   # 2 in-channels: image + t
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),              # 1 out-channel: velocity
        )

    def forward(self, x, t):
        # t arrives as (batch,1,1,1); expand to a constant 28x28 grid so it can be
        # stacked as a second channel — this is how every pixel learns the time.
        t = t.expand(-1, 1, 28, 28)
        return self.net(torch.cat([x, t], dim=1))

model = ConvNet().to(device)   # .to(device) puts weights on the GPU

# 3. Loss
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 4. Train
for epoch in range(10):
    total = 0
    for x, _ in train_loader:
        x = x.to(device)
        noise = torch.randn_like(x)
        t = torch.rand(x.shape[0], 1, 1, 1, device=device)
        x_t = (1-t)*noise + t*x
        optimizer.zero_grad()
        outputs = model(x_t, t)
        loss = criterion(outputs, x - noise)
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch}, Loss: {total / len(train_loader):.4f}")

# 5. Euler sampler
x = torch.randn(16, 1, 28, 28, device=device)
steps = 100
dt = 1.0 / steps
with torch.no_grad():
    for i in range(steps):
        t = torch.full((16, 1, 1, 1), i * dt, device=device)
        x = x + model(x, t) * dt

# 6. Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for img, ax in zip(x.cpu(), axes.flat):
    ax.imshow(img.view(28, 28), cmap='gray')
    ax.axis('off')
plt.show()
