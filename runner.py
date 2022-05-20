from autoregressivePixelRCNN import main
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

train_data = sys.argv[1]
test_data = sys.argv[2]
shape_h = int(sys.argv[3])
shape_w = int(sys.argv[4])
shape_c = int(sys.argv[5])
epochs = int(sys.argv[6])
lr = float(sys.argv[7])
num_samples = int(sys.argv[8])
save_dir = sys.argv[9]

shape = (shape_h, shape_w, shape_c)

print("Loading data.")
train_data = np.load(train_data)[:20]
test_data = np.load(test_data)[:20]

print("Training.")
train_loss, test_loss, samples, model, optimizer = main(train_data, test_data, shape, epochs, lr, num_samples)

torch.save(
    {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "test_loss": test_loss,
    },
    os.path.join(save_dir, "model.pt"),
)

np.save(os.path.join(save_dir, "samples.npy"), samples)

plt.figure(figsize=(8, 6), dpi=100)
ax = plt.subplot()
plt.plot(train_loss, label="Train Loss")
plt.plot(test_loss, label="Test Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "loss.jpg"))