import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def get_cifar10():
    from torchvision.datasets import CIFAR10

    root_dir = r"/Users/langong/Projects/diffusion/datasets/cifar10"
    train_dir =  os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")
    trainset = CIFAR10(root=root_dir, train=True, download=True)
    testset = CIFAR10(root=root_dir, train=False, download=True)

    # Re-organize the downloaded image binaries to directory format suitable for ImageFolder
    def save_images(dataset, data_dir):
        for idx, (img, label) in enumerate(dataset):
            label_dir = os.path.join(data_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            img_path = os.path.join(label_dir, f"{idx}.png")
            img.save(img_path)

    save_images(trainset, train_dir)
    save_images(testset, test_dir)
