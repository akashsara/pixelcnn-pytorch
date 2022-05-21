import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from masked_cnn_layer import MaskedConv2d, AutoregressiveMaskedConv2d, AutoregressiveResidualMaskedConv2d


class AutoregressiveColorPixelRCNN(nn.Module):
    """ Pixel Residual-CNN-class using residual blocks as shown in figure 5 from "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al. """
    def __init__(self, in_channels, out_channels, conv_filters, device):
        super().__init__()
        self.net = nn.Sequential(
            # A 7x7 A-type convolution
            MaskedConv2d('A', in_channels, conv_filters, kernel_size=7, padding=3), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            # 8 type-B residual convolutons
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveMaskedConv2d('B', conv_filters, out_channels, kernel_size=1, padding=0)
        ).to(device)

    def forward(self, x):
        return self.net(x)


def main(train_data, test_data, image_shape, epochs=10, lr=1e-3, batch_size=128):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in [0, 255]
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in [0, 255]
    image_shape: (H, W, C), height, width, and # of channels of the image

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    """
    H, W, C = image_shape
    output_bits = 256

    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")

    def cross_entropy_loss(batch, output):
        per_bit_output = output.reshape(batch.shape[0], output_bits, C, H, W)
        return torch.nn.CrossEntropyLoss()(per_bit_output, batch.long())

    def get_test_loss(dataloader, model):
        test_loss = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader)):
                out = model(batch.to(device))
                loss = cross_entropy_loss(batch, out)
                test_loss.append(loss.item())

        return np.mean(np.array(test_loss))

    num_dataloader_workers = 4 if gpu else 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_dataloader_workers,
    pin_memory=gpu)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_dataloader_workers,
    pin_memory=gpu)

    no_channels, out_channels, convolution_filters = C, C * output_bits, 120
    pixelrcnn_auto = AutoregressiveColorPixelRCNN(no_channels, out_channels, convolution_filters, device).to(device)

    optimizer = torch.optim.Adam(pixelrcnn_auto.parameters(), lr=lr)

    train_losses = []
    test_losses = [get_test_loss(test_loader, pixelrcnn_auto)]

    # Training
    for epoch in tqdm(range(epochs)):
        for _, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = pixelrcnn_auto(batch.to(device))
            loss = cross_entropy_loss(batch, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        test_loss = get_test_loss(test_loader, pixelrcnn_auto)
        test_losses.append(test_loss)
        print(f"Epoch: [{epoch + 1}/{epochs}] Train Loss: {loss} \tTest Loss: {test_loss}")
   
    return np.array(train_losses), np.array(test_losses), pixelrcnn_auto, optimizer

def sample(num_samples, shape, model):
    """
    num_samples: int, number of samples to generate
    image_shape: (H, W, C), height, width, and # of channels of the image
    model: trained model

    Returns
    - a numpy array of size (num_samples, H, W, C) of samples with values in [0, 255]
    """
    H, W, C = shape
    output_bits = 256

    def get_proba(output):
        return torch.nn.functional.softmax(output.reshape(output.shape[0], output_bits, C, H, W), dim=1)

    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")

    model.to(device)
    model.eval()
    if gpu:
        torch.cuda.empty_cache()

    # Sampling
    samples = torch.zeros(size=(num_samples, C, H, W)).to(device)
    with torch.no_grad():
        for i in tqdm(range(H)):
            for j in range(W):
                for c in range(C):
                    out = model(samples)
                    proba = get_proba(out)
                    samples[:, c, i, j] = torch.multinomial(proba[:, :, c, i, j], 1).squeeze().float()
    return np.transpose(samples.detach().cpu().numpy(), [0, 2, 3, 1])