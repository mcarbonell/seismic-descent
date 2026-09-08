import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from seismic_optimizer import SeismicOptimizer
import time

try:
    import torch_directml
    device = torch_directml.device()
    print("Using device:", device)
except ImportError:
    device = torch.device("cpu")
    print("Using device: CPU")

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

def train_and_eval(name, optimizer, model, train_loader, test_loader, epochs=20, use_adaptive=False):
    print(f"\n--- Training {name} ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            if use_adaptive:
                optimizer.step(loss=loss.item())
            else:
                optimizer.step()
                
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch:2d}: Train Loss={avg_train_loss:.4f}, Test Accuracy={acc:.2f}%")
        
    return acc

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

    epochs = 20

    # 1. Adaptive Floored Seismic
    model_seismic = SimpleMLP().to(device)
    opt_seismic = SeismicOptimizer(
        model_seismic.parameters(), 
        lr=0.2, 
        noise_amplitude=0.01, 
        adaptive_power=2.0, 
        adaptive_floor=0.2
    )
    acc_seismic = train_and_eval("Adaptive Floored Seismic", opt_seismic, model_seismic, train_loader, test_loader, epochs, use_adaptive=True)

    # 2. SGD (Reference)
    model_sgd = SimpleMLP().to(device)
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.2)
    acc_sgd = train_and_eval("SGD (Reference)", opt_sgd, model_sgd, train_loader, test_loader, epochs, use_adaptive=False)

    # 3. Adam (Reference)
    model_adam = SimpleMLP().to(device)
    opt_adam = optim.Adam(model_adam.parameters(), lr=0.001)
    acc_adam = train_and_eval("Adam (Reference)", opt_adam, model_adam, train_loader, test_loader, epochs, use_adaptive=False)

    print("\n" + "="*45)
    print(f"{'Optimizer':<25} | {'Final Accuracy':<15}")
    print("-" * 45)
    print(f"{'Adaptive Floored Seismic':<25} | {acc_seismic:.2f}%")
    print(f"{'SGD':<25} | {acc_sgd:.2f}%")
    print(f"{'Adam':<25} | {acc_adam:.2f}%")
    print("="*45)

if __name__ == '__main__':
    main()
