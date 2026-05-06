import os
import numpy as np
import torch
from torch.utils.data import Dataset

class OPERAnetDataset(Dataset):
    def __init__(self, root_dir, subjects_to_include=None, window_size=340, step_size=170, transform=None):
        """
        subjects_to_include: List of strings like ['S1', 'S2', 'S5'] 
                             If None, includes all subjects.
        """
        self.samples = []
        self.transform = transform
        self.label_map = {'W': 0, 'S': 1, 'T': 2, 'L': 3, 'B': 4, 'F': 5, 'X': 6}

        print(f"Loading data for subjects: {subjects_to_include if subjects_to_include else 'ALL'}...")
        
        for root, _, files in os.walk(root_dir):
            # Extract Subject ID from folder name (e.g., 'S7a' -> 'S7')
            folder_name = os.path.basename(root)
            subject_id = folder_name[:2] # Gets 'S1', 'S7', etc.

            # Step 1: Logic for Subject-Based Splitting
            if subjects_to_include is not None and subject_id not in subjects_to_include:
                continue

            for file in files:
                if file.endswith(".txt"):
                    try:
                        code = file.split('_')[1].upper()
                        if code in self.label_map:
                            file_path = os.path.join(root, file)
                            data = np.load(file_path, allow_pickle=True).astype(np.float32)
                            length = data.shape[0]
                            label = self.label_map[code]
                            
                            for start in range(0, length - window_size + 1, step_size):
                                end = start + window_size
                                window = data[start:end, :] 
                                self.samples.append((window, label))
                    except Exception:
                        continue
        
        print(f"Loaded {len(self.samples)} windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, label = self.samples[idx]
        
        # Apply Data Augmentation (Step 2)
        if self.transform:
            window = self.transform(window)
        
        tensor_x = torch.from_numpy(window).unsqueeze(0) # (1, 340, 100)
        tensor_y = torch.tensor(label, dtype=torch.long)
        return tensor_x, tensor_y

import random

class WirelessAugment:
    def __init__(self, time_mask_max=30, freq_mask_max=10, p=0.5):
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.p = p # Probability of applying augmentation

    def __call__(self, x):
        """
        x: numpy array of shape (340, 100) -> (Time, Velocity/Freq)
        """
        if random.random() > self.p:
            return x

        # 1. Add Gaussian Noise (Simulates hardware thermal noise)
        noise = np.random.normal(0, 0.01, x.shape).astype(np.float32)
        x = x + noise

        # 2. Time Masking (Simulates packet loss / brief signal blockage)
        # Masks a vertical strip across all velocity bins
        t = np.random.randint(0, self.time_mask_max)
        t0 = np.random.randint(0, x.shape[0] - t)
        x[t0:t0+t, :] = 0

        # 3. Frequency Masking (Simulates static multipath interference)
        # Masks a horizontal strip across all time frames
        f = np.random.randint(0, self.freq_mask_max)
        f0 = np.random.randint(0, x.shape[1] - f)
        x[:, f0:f0+f] = 0

        # 4. Global Normalization (Crucial for Environment Independence)
        # Normalizes the window so the model looks at movement patterns, not raw power
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        return x
    
class MultiAntennaTestDataset(Dataset):
    def __init__(self, root_dir, window_size=340, step_size=170):
        self.samples = []
        self.label_map = {
            'W': 0, 'R': 1, 'J': 2, 'L': 3, 'S': 4, 'C': 5, 'H': 6, 'G': 6
        }

        print(f"Loading multi-antenna files from {root_dir}...")
        for root, _, files in os.walk(root_dir):
            for file in files:
                # Only look for stream 1, we will automatically load the others
                if file.endswith("stream_1.txt"):
                    try:
                        parts = file.split('_')
                        if len(parts) < 2:
                            continue
                        
                        # FIX 1: Extract first character to handle suffixes like 'J3' -> 'J'
                        code = parts[1][0].upper()
                        if code in self.label_map:
                            label = self.label_map[code]
                            base_filename = file.replace("_stream_1.txt", "")
                            base_path = os.path.join(root, base_filename)
                            
                            # Load all 4 streams to ensure they are the exact same length
                            streams = []
                            for i in range(1, 5):
                                stream_file = f"{base_path}_stream_{i}.txt"
                                
                                # FIX 2: If stream 4 (or any stream) is missing on disk, fallback to stream 1
                                if not os.path.exists(stream_file):
                                    stream_file = f"{base_path}_stream_1.txt"
                                
                                streams.append(np.load(stream_file, allow_pickle=True).astype(np.float32))
                            
                            length = streams[0].shape[0]
                            
                            # Slice into 340-frame windows
                            for start in range(0, length - window_size + 1, step_size):
                                end = start + window_size
                                self.samples.append(([s[start:end, :] for s in streams], label))
                    except Exception as e:
                        continue
                        
        print(f"Successfully loaded {len(self.samples)} multi-antenna windows!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        streams, label = self.samples[idx]
        
        processed_tensors = []
        for s in streams:
            # Z-Score Standardization: (x - mean) / std
            # This makes Subject 7's signal look exactly like Subject 1's signal scale
            s_norm = (s - np.mean(s)) / (np.std(s) + 1e-8)
            processed_tensors.append(torch.from_numpy(s_norm).unsqueeze(0))
        
        return processed_tensors[0], processed_tensors[1], \
               processed_tensors[2], processed_tensors[3], \
               torch.tensor(label, dtype=torch.long)
    
# --- 1. Define Subject Groups for Independent Evaluation ---
# S1-S5: Training (Multiple environments/people)
# S6: Validation (Checking performance on a new person during training)
# S7: Testing (The ultimate "Independent" test as per the paper)
train_subs = ['S1', 'S2', 'S3', 'S4', 'S5']
val_subs   = ['S6']
test_subs  = ['S7']

# --- 2. Create a "Normalization Only" transform for Val/Test ---
# We want Val/Test to be normalized like Train, but NOT masked/noisy.
class NormalizeOnly:
    def __call__(self, x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)

# --- 3. Initialize the Datasets Independently ---
# Note: We apply WirelessAugment ONLY to the training set.
train_dataset = OPERAnetDataset(
    root_dir="doppler_traces", 
    subjects_to_include=train_subs, 
    transform=WirelessAugment(p=0.5) # Apply noise/masking
)

val_dataset = OPERAnetDataset(
    root_dir="doppler_traces", 
    subjects_to_include=val_subs, 
    transform=NormalizeOnly() # Only scale the data
)

test_dataset = OPERAnetDataset(
    root_dir="doppler_traces", 
    subjects_to_include=test_subs, 
    transform=NormalizeOnly() # Only scale the data
)

# --- 4. Create the DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"--- Dataset Split Summary ---")
print(f"Train (S1-S5): {len(train_dataset)} windows | Batches: {len(train_loader)}")
print(f"Val   (S6):    {len(val_dataset)} windows | Batches: {len(val_loader)}")
print(f"Test  (S7):    {len(test_dataset)} windows | Batches: {len(test_loader)}")

class BaseLineModel(nn.Module):
    def __init__(self, num_classes = 7):

        super().__init__()
        
        self.branch1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=2, stride=2),
                                     nn.ReLU()
                                     )
        
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1, padding="same"),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=6, out_channels=9, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU()
                                     )
        
        self.concat_conv = nn.Sequential(nn.Conv2d(in_channels=15, out_channels=3, kernel_size=1, stride=1),
                                         nn.ReLU())
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(3 * 170 * 50, num_classes)
    
    def forward(self, x):

        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        x = torch.cat([out1, out2, out3], dim=1)

        x = self.concat_conv(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)

        return x
    
def sharp_decision_fusion(logits_list):
    """
    logits_list: A list of 4 tensors, each of shape [batch_size, num_classes]
    Returns: A tensor of final predicted labels of shape [batch_size]
    """
    # Stack into shape: [4, batch_size, num_classes]
    stacked_logits = torch.stack(logits_list) 
    
    # Get individual predictions: Shape [4, batch_size]
    stacked_preds = torch.argmax(stacked_logits, dim=2) 
    
    final_preds = []
    batch_size = stacked_preds.shape[1]
    
    # Iterate through each sample in the batch
    for i in range(batch_size):
        preds_for_sample = stacked_preds[:, i].tolist()
        
        # Count the votes
        counts = Counter(preds_for_sample)
        most_common_pred, count = counts.most_common(1)[0]
        
        # SHARP Rule: If at least 3 out of 4 antennas agree
        if count >= 3:
            final_preds.append(most_common_pred)
        else:
            # Tie-breaker: Sum the raw logits across all 4 antennas for this sample
            summed_logits = torch.sum(stacked_logits[:, i, :], dim=0) # Shape: [num_classes]
            final_preds.append(torch.argmax(summed_logits).item())
            
    return torch.tensor(final_preds, device=logits_list[0].device)

import torch.optim as optim
from tqdm import tqdm # Optional: for progress bars

# 1. Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Initialize Model, Loss, and Optimizer
model = BaseLineModel(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# To help the model converge better on augmented data
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}

best_val_acc = 0.0
num_epochs = 5 # Increase this if using Augmentation, as it takes longer to learn

print("Starting Training...")

for epoch in range(num_epochs):
    # --- TRAINING PHASE ---
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    # Using tqdm for a nice progress bar
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
    for batch_x, batch_y in loop:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
        
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total
    
    # Update Learning Rate Scheduler
    scheduler.step(epoch_val_acc)

    # --- RECORD HISTORY (Fixed placement) ---
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_acc'].append(epoch_val_acc)

    # Save Best Model Weights
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), 'best_sharp_model.pth')
        print(f"--> Best model saved with Val Acc: {best_val_acc:.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}] Summary: "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

# --- 4. FINAL TESTING PHASE WITH FUSION ---

# IMPORTANT: Load the BEST weights found during training before testing
model.load_state_dict(torch.load('best_sharp_model.pth'))
model.eval()

# Ensure MultiAntennaTestDataset also uses normalization
test_dataset_fusion = MultiAntennaTestDataset(root_dir="doppler_traces/S7a")
test_loader_fusion = DataLoader(test_dataset_fusion, batch_size=64, shuffle=False)

print("\nRunning SHARP Decision Fusion on Subject 7 (Unseen)...")
test_correct, test_total = 0, 0

with torch.no_grad():
    for x1, x2, x3, x4, labels in test_loader_fusion:
        # Move all antennas and labels to device
        x1, x2, x3, x4 = x1.to(device), x2.to(device), x3.to(device), x4.to(device)
        labels = labels.to(device)
        
        # Optional: Manual Normalization if not in Dataset __getitem__
        # x1 = (x1 - x1.mean()) / (x1.std() + 1e-8) ... (repeat for all)

        logits = [model(x1), model(x2), model(x3), model(x4)]
        
        # Apply Fusion
        final_predictions = sharp_decision_fusion(logits)
        
        # Ensure predictions are on the same device as labels for comparison
        final_predictions = final_predictions.to(device)
        
        test_total += labels.size(0)
        test_correct += (final_predictions == labels).sum().item()

test_acc = test_correct / test_total
print(f"\nFINAL RESULT:")
print(f"SHARP Fusion Test Accuracy (Subject 7): {test_acc:.4f} ({test_acc * 100:.2f}%)")