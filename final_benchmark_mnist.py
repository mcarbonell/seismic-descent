import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from seismic_optimizer import SeismicOptimizer
import time
import matplotlib.pyplot as plt

# Setup device
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

def train_and_eval(name, optimizer, model, train_loader, test_loader, epochs=10):
    print(f"\n--- Training {name} ---")
    train_losses = []
    test_accs = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
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
        test_accs.append(acc)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={acc:.2f}%")
        
    return train_losses, test_accs

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

    epochs = 10

    # 1. Seismic Optimizer (Best Params)
    model_seismic = SimpleMLP().to(device)
    opt_seismic = SeismicOptimizer(
        model_seismic.parameters(), 
        lr=0.1, 
        noise_amplitude=0.1, 
        noise_decay=0.999
    )
    hist_seismic_loss, hist_seismic_acc = train_and_eval("Seismic", opt_seismic, model_seismic, train_loader, test_loader, epochs)

    # 2. SGD (Reference)
    model_sgd = SimpleMLP().to(device)
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.1)
    hist_sgd_loss, hist_sgd_acc = train_and_eval("SGD", opt_sgd, model_sgd, train_loader, test_loader, epochs)

    # 3. Adam (Reference)
    model_adam = SimpleMLP().to(device)
    opt_adam = optim.Adam(model_adam.parameters(), lr=0.001)
    hist_adam_loss, hist_adam_acc = train_and_eval("Adam", opt_adam, model_adam, train_loader, test_loader, epochs)

    print("\n" + "="*40)
    print(f"{'Optimizer':<15} | {'Final Acc':<10}")
    print("-" * 40)
    print(f"{'Seismic':<15} | {hist_seismic_acc[-1]:.2f}%")
    print(f"{'SGD':<15} | {hist_sgd_acc[-1]:.2f}%")
    print(f"{'Adam':<15} | {hist_adam_acc[-1]:.2f}%")
    print("="*40)

if __name__ == '__main__':
    main()
