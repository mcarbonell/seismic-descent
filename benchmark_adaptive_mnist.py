import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from seismic_optimizer import SeismicOptimizer
import time

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

def train_adaptive(model, optimizer, train_loader, use_adaptive=True):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Pass loss to optimizer if adaptive
        if use_adaptive:
            optimizer.step(loss=loss.item())
        else:
            optimizer.step()
            
        total_loss += loss.item()
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

    print(f"{'Method':<20} | {'LR':<6} | {'Base Amp':<8} | {'Avg Loss'}")
    print("-" * 55)

    # 1. Fixed Seismic (Current Best)
    model_fixed = SimpleMLP().to(device)
    opt_fixed = SeismicOptimizer(model_fixed.parameters(), lr=0.2, noise_amplitude=0.005)
    loss_fixed = train_adaptive(model_fixed, opt_fixed, train_loader, use_adaptive=False)
    print(f"{'Fixed Seismic':<20} | {0.2:<6} | {0.005:<8} | {loss_fixed:.4f}")

    # 2. Adaptive Seismic (Linear)
    model_adapt = SimpleMLP().to(device)
    opt_adapt = SeismicOptimizer(model_adapt.parameters(), lr=0.2, noise_amplitude=0.025, adaptive_power=1.0)
    loss_adapt = train_adaptive(model_adapt, opt_adapt, train_loader, use_adaptive=True)
    print(f"{'Adaptive Linear':<20} | {0.2:<6} | {0.025:<8} | {loss_adapt:.4f}")

    # 3. Adaptive Seismic (Quadratic)
    model_quad = SimpleMLP().to(device)
    opt_quad = SeismicOptimizer(model_quad.parameters(), lr=0.2, noise_amplitude=0.01, adaptive_power=2.0)
    loss_quad = train_adaptive(model_quad, opt_quad, train_loader, use_adaptive=True)
    print(f"{'Adaptive Quad':<20} | {0.2:<6} | {0.01:<8} | {loss_quad:.4f}")

    # 4. Adaptive Seismic (Floored Quad)
    model_floor = SimpleMLP().to(device)
    opt_floor = SeismicOptimizer(model_floor.parameters(), lr=0.2, noise_amplitude=0.01, adaptive_power=2.0, adaptive_floor=0.2)
    loss_floor = train_adaptive(model_floor, opt_floor, train_loader, use_adaptive=True)
    print(f"{'Adaptive Floored':<20} | {0.2:<6} | {0.01:<8} | {loss_floor:.4f}")

    # 5. SGD Reference
    model_sgd = SimpleMLP().to(device)
    opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.2)
    loss_sgd = train_adaptive(model_sgd, opt_sgd, train_loader, use_adaptive=False)
    print(f"{'SGD Reference':<20} | {0.2:<6} | {'-':<8} | {loss_sgd:.4f}")

if __name__ == '__main__':
    main()
