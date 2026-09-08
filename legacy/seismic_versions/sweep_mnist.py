import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from seismic_optimizer import SeismicOptimizer
import time
import itertools

# Setup device
try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    device = torch.device("cpu")

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_one_epoch(model, optimizer, train_loader):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Early stop for sweep speed: use first 200 batches
        if batch_idx >= 200:
            break
    return total_loss / 200

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # Grid search parameters - Limit tests
    lrs = [0.15, 0.2]
    amps = [0.005, 0.01]
    decays = [0.999]
    
    results = []

    print(f"{'LR':<8} | {'Amp':<8} | {'Decay':<8} | {'Loss':<8}")
    print("-" * 45)

    for lr, amp, decay in itertools.product(lrs, amps, decays):
        model = SimpleMLP().to(device)
        optimizer = SeismicOptimizer(
            model.parameters(), 
            lr=lr, 
            noise_amplitude=amp, 
            noise_decay=decay,
            n_cycles=5
        )
        
        loss = train_one_epoch(model, optimizer, train_loader)
        results.append((lr, amp, decay, loss))
        print(f"{lr:<8} | {amp:<8} | {decay:<8} | {loss:<8.4f}")

    # Add SGD Baselines
    print("\n--- SGD Baselines ---")
    for lr in lrs:
        model = SimpleMLP().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss = train_one_epoch(model, optimizer, train_loader)
        print(f"SGD (LR={lr:<4}) | Loss={loss:.4f}")

    # Find best
    results.sort(key=lambda x: x[3])
    best = results[0]
    print("\n" + "="*45)
    print(f"BEST CONFIG: LR={best[0]}, Amp={best[1]}, Decay={best[2]} | Loss={best[3]:.4f}")
    print("="*45)

if __name__ == '__main__':
    main()
