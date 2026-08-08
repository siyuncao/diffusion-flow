import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from data_loader_pytorch import X_traintensor, y_traintensor, X_testtensor, y_testtensor, dataloader

# 1. load
data = load_breast_cancer()
X = data.data
y = data.target

# 2. split (before normalizing, to avoid leaking test info into train stats)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# 3. normalize (stats computed from train only, applied to both)
mean_x = np.mean(X_train_raw, axis=0)
std_X = np.std(X_train_raw, axis=0)

X_train = (X_train_raw - mean_x) / std_X
X_test = (X_test_raw - mean_x) / std_X

# 4. tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 5. datasets + loaders (shuffle only needed for training)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# 6. build the model
model = nn.Sequential(
    nn.Linear(785, 18),   # in_features: 30 columns — each column is one feature (mean radius, mean texture, etc.)
    # out_features: it's your choice, feel free to trail and error
    nn.ReLU(),         # activation between layers
    nn.Linear(18, 784),   # out_features: 1 feature is enough --> have cancer (eg. >30%) or no cancer (eg. <30%)
    nn.Sigmoid()       # corresponds to sigmoid formula typed out in numpy version
)

# 7. loss function
criterion = nn.BCELoss()

# optimizer (corresponds to update parameters in numpy version)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 8. training loop
for epoch in range(100):
    optimizer.zero_grad() # PyTorch accumulates gradients by default, this line is to clear them before each update
    outputs = model(X_traintensor) # corresponds to z1, a1, z2, a2 = forward(...) in numpy version
    loss = criterion(outputs, y_traintensor.unsqueeze(1)) # corresponds to L = loss_function(...) in numpy version
    loss.backward() # corresponds to dW1, db1, dW2, db2 = backward(...) in numpy version
    optimizer.step() # corresponds to W1, b1, W2, b2 = update_params(...) in numpy version
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 9. evaluation 
model.eval()
with torch.no_grad(): # a Python context manager: runs once, sets something up, and tears it down cleanly at the end
    test_outputs = model(X_testtensor)
    predictions = (test_outputs >= 0.5).float()
    accuracy = (predictions == y_testtensor.unsqueeze(1)).float().mean()
    print(f"Test accuracy: {accuracy:.4f}")
