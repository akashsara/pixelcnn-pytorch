import autoregressivePixelRCNN as PixelCNN
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import data
from tqdm import tqdm
from PIL import Image

train_data = sys.argv[1]
test_data = sys.argv[2]
shape_h = int(sys.argv[3])
shape_w = int(sys.argv[4])
shape_c = int(sys.argv[5])
epochs = int(sys.argv[6])
lr = float(sys.argv[7])
batch_size = int(sys.argv[8])
output_batch_size = int(sys.argv[9])
output_num_batches = int(sys.argv[10])
save_dir = sys.argv[11]

shape = (shape_h, shape_w, shape_c)

print("Loading data.")
# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(64)

train_data = data.CustomDatasetNoMemory(train_data, transform, use_noise_images=False)
test_data = data.CustomDatasetNoMemory(test_data, transform, use_noise_images=False)

print("Training.")
train_loss, test_loss, model, optimizer = PixelCNN.main(train_data, test_data, shape, epochs, lr, batch_size)

print("Training complete. Saving.")

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

plt.figure(figsize=(8, 6), dpi=100)
ax = plt.subplot()
plt.plot(train_loss, label="Train Loss")
plt.plot(test_loss, label="Test Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "loss.jpg"))

print("Training complete. Sampling.")
all_samples = []
for i in tqdm(range(output_num_batches)):
    samples = PixelCNN.sample(output_batch_size, shape, model)
    all_samples.append(samples)

print("Saving samples.")
np.save(os.path.join(save_dir, "samples.npy"), samples)
for i, image in enumerate(all_samples):
    image = Image.fromarray(image)
    image.save(os.path.join(save_dir, f"{i}.png"))