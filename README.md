<<<<<<< HEAD
## 🧠 The Standard PyTorch Deep Learning Workflow

Every PyTorch project, whether it's recognizing digits or generating text, follows this exact same 7-step pipeline.

### 1. Create the Dataset Class
If you aren't using a pre-built dataset (like `torchvision.datasets`), you need to define how PyTorch loads your data. It must inherit from `torch.utils.data.Dataset` and need these three methods:
```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data) # Returns total number of samples

    def __getitem__(self, idx):
        # Fetches one sample and its label at a time
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
```

### 2. Wrap it in a DataLoader
The DataLoader handles batching, shuffling, and multi-process data loading.

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### 3. Define the model structure
Your network must inherit from nn.Module. Define layers in __init__ and data flow in forward.

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# Initialize and move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleModel().to(device)
```

### 4. Choose loss function and optimizer
- Loss: Measures how "wrong" the model's predictions are.
- Optimizer: Updates the model's weights to minimize the loss.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss() # Standard for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 5.The Training Loop
The core process where learning happens. The 5-step sequence inside the batch loop is mandatory.
```python
epochs = 5

for epoch in range(epochs):
    model.train() # Set to training mode
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. Zero gradients (clear previous calculations)
        optimizer.zero_grad()
        
        # 2. Forward pass (make predictions)
        outputs = model(inputs)
        
        # 3. Calculate loss
        loss = criterion(outputs, labels)
        
        # 4. Backward pass (calculate gradients)
        loss.backward()
        
        # 5. Optimizer step (update weights)
        optimizer.step()
    
    print(f"Epoch {epoch+1} complete.")
```

### 6. Evaluation Loop
Check performance on unseen data. Always use torch.no_grad() to save memory and disable gradient calculations.
```python
model.eval() # Set to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        # Get the index of the highest probability
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
```

### 7.Save and Load the Model
Save only the learned weights (state_dict) rather than the entire object for better compatibility.
```python
# --- Save ---
torch.save(model.state_dict(), "model_weights.pth")

# --- Load ---
model = SimpleModel() # Must recreate the architecture first
model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
model.to(device)
model.eval()
```
>>>>>>> 9aa47a80e00658fa412a6153f947bdc1b69f99ad
