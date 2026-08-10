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

# 2. UNet
# Flat conv stack gave each output pixel only a 9x9 view — good local strokes,
# no global coherence. Downsampling means the middle layers see the whole
# image; skip connections carry fine detail back to the output.
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(),
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = block(2, 64)      # 28x28
        self.down2 = block(64, 128)    # 14x14
        self.mid   = block(128, 256)   # 7x7  — sees the whole image
        self.up2   = block(256 + 128, 128)
        self.up1   = block(128 + 64, 64)
        self.out   = nn.Conv2d(64, 1, 1)
        self.pool  = nn.MaxPool2d(2)
        self.up    = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):
        t = t.expand(-1, 1, 28, 28)
        d1 = self.down1(torch.cat([x, t], dim=1))   # 28
        d2 = self.down2(self.pool(d1))              # 14
        m  = self.mid(self.pool(d2))                # 7
        u2 = self.up2(torch.cat([self.up(m), d2], dim=1))    # 14
        u1 = self.up1(torch.cat([self.up(u2), d1], dim=1))   # 28
        return self.out(u1)

model = UNet().to(device)

# 3. Loss
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 4. Train
for epoch in range(20):
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
