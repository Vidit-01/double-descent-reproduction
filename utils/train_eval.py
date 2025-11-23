import torch
from torch import nn

def train_model(model, train_loader, lr=1e-3, epochs=50, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb.view(xb.size(0), -1))
            loss = criterion(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

def eval_model(model, loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb.view(xb.size(0), -1))
            loss_total += criterion(out, yb).item() * xb.size(0)
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

    return loss_total / total, 1 - correct / total
