import torch
import numpy as np
import matplotlib.pyplot as plt

from data.mnist import get_mnist
from models.mlp import make_mlp
from utils.train_eval import train_model, eval_model

device = "cuda" if torch.cuda.is_available() else "cpu"

widths = [2,4,8,16,32,64,128,256,512,1024,2048,4096]

train_errs, test_errs = [], []

train_loader, test_loader = get_mnist(train_size=3000, noise_rate=0.2)

for w in widths:
    torch.manual_seed(0)
    model = make_mlp(width=w)
    train_model(model, train_loader, lr=1e-3, epochs=30, device=device)
    tr_loss, tr_err = eval_model(model, train_loader, device=device)
    ts_loss, ts_err = eval_model(model, test_loader, device=device)
    train_errs.append(tr_err)
    test_errs.append(ts_err)

plt.plot(widths, train_errs, marker="o", label="Train Error")
plt.plot(widths, test_errs, marker="o", label="Test Error")
plt.xscale("log")
plt.legend()
plt.savefig("../plots/width_sweep.png")
