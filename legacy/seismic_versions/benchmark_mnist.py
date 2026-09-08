import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from seismic_optimizer import SeismicOptimizer
import time

# Attempt to use torch-directml for AMD Radeon GPU
try:
    import torch_directml
    device = torch_directml.device()
    print("Using torch-directml device:", device)
except ImportError:
    device = torch.device("cpu")
    print("torch-directml not found, using CPU")

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

def train(model, optimizer, train_loader, epochs=2):
    model.train()
    history = []
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
            history.append(loss.item())
    return history

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # 1. Train with Seismic Descent
    print("\n--- Training with SeismicOptimizer ---")
    model_seismic = SimpleMLP().to(device)
    # n_cycles = 5 for a short test
    optimizer_seismic = SeismicOptimizer(
        model_seismic.parameters(), 
        lr=0.01, 
        noise_amplitude=0.5, # Slightly lower for NN stability
        n_cycles=5
    )
    start_t = time.time()
    hist_seismic = train(model_seismic, optimizer_seismic, train_loader, epochs=1)
    print(f"Seismic Training took: {time.time() - start_t:.2f}s")

    # 2. Train with Adam (Comparison)
    print("\n--- Training with Adam ---")
    model_adam = SimpleMLP().to(device)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
    start_t = time.time()
    hist_adam = train(model_adam, optimizer_adam, train_loader, epochs=1)
    print(f"Adam Training took: {time.time() - start_t:.2f}s")

    # Output final summary
    print("\nBenchmark Complete.")
    print(f"Final Loss (Seismic): {hist_seismic[-1]:.4f}")
    print(f"Final Loss (Adam):    {hist_adam[-1]:.4f}")

if __name__ == '__main__':
    main()
