import torch
from torchvision import transforms
from PIL import Image
import os

class CustomDatasetNoMemory(torch.utils.data.Dataset):
    """
    Requires the path to a dataset.
    Essentially the same as above but it doesn't load all the data to memory.
    Returns filename, image.
    """

    def __init__(self, dataset_directory, transform, use_noise_images):
        self.dataset_path = dataset_directory
        all_images = os.listdir(dataset_directory)
        if not use_noise_images:
            all_images = [x for x in all_images if "noise" not in x]
        self.all_images = all_images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.all_images[index]
        image_path = os.path.join(self.dataset_path, filename)
        image = Image.open(image_path).convert("RGB")
        processed = self.transform(image)
        return processed

    def __len__(self):
        return len(self.all_images)

def image2tensor_resize(image_size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]
    )