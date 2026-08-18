import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load
train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([ # Compose is added first in order to chain the latter (ToTensor, Normalize) together in sequence
    transforms.ToTensor(), # The MNIST file on disk stores 784 bytes per digit. 
                           # ToTensor() reads those bytes into a PyTorch tensor and divides by 255, mapping everything into [0, 1].
    transforms.Normalize((0.5,), (0.5,)) # maps [0, 1] → [-1, 1]
]))

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# 2. UNet model
def helper(cin, cout): # why helper()? you need this same pattern five times (down1, down2, mid, up2, up1). 
                      # Without it you'd write four lines each, five times over.
  return nn.Sequential(
      nn.Conv2d(cin, cout, 3, padding=1), 
      nn.ReLU(),
      nn.Conv2d(cout, cout, 3, padding=1), 
      nn.ReLU(),
  )

class UNet(nn.Module):
  def __init__(self): # creates the layers (weights get allocated once, at construction)
    super().__init__()
    self.down1 = helper(2, 64)
    self.down2 = helper(64, 128)
    self.bottleneck = helper(128, 256)
    self.up1        = helper(256 + 128, 128) # 14x14, +d2 skip
    self.up2        = helper(128 + 64, 64)   # 28x28, +d1 skip
    self.out        = nn.Conv2d(64, 1, 1)
    self.pool       = nn.MaxPool2d(2)
    self.up         = nn.Upsample(scale_factor=2, mode='nearest')
    
  def forward(self, x, t): # uses them, in order, every time data passes through
    t  = t.expand(-1, 1, 28, 28)
    d1 = self.down1(torch.cat([x, t], dim=1))
    d2 = self.down2(self.pool(d1))
    m  = self.bottleneck(self.pool(d2))                # 7
    u1 = self.up1(torch.cat([self.up(m), d2], dim=1))  # 14
    u2 = self.up2(torch.cat([self.up(u1), d1], dim=1)) # 28
    return self.out(u2)

model = UNet().to(device)

# 3. noise schedule
T = 1000
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

# 4. Loss
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Train
for epoch in range(20):
    total = 0
    for x, _ in train_loader:
        x = x.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(0, T, (x.shape[0],), device=device)
        ab = alpha_bar[t].view(-1, 1, 1, 1)
        x_t = ab.sqrt() * x + (1 - ab).sqrt() * noise
        optimizer.zero_grad()
        outputs = model(x_t, (t / T).view(-1, 1, 1, 1))
        loss = criterion(outputs, noise)
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch}, Loss: {total / len(train_loader):.4f}")

# 6. DDPM sampler
x = torch.randn(16, 1, 28, 28, device=device)
with torch.no_grad():
    for i in reversed(range(T)):
        t  = torch.full((16,), i, device=device)
        tf = (t.float() / T).view(-1, 1, 1, 1)
        eps = model(x, tf)

        a, ab = alphas[i], alpha_bar[i]
        x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()

        if i > 0:
            x = x + betas[i].sqrt() * torch.randn_like(x)
            
# 7. Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for img, ax in zip(x.cpu(), axes.flat):
    ax.imshow(img.view(28, 28), cmap='gray')
    ax.axis('off')
plt.show()
