import torch
from torchvision import datasets, transforms

__all__ = ['get_dataloaders']

def get_dataloaders(cfg):
    # Define transform
    transform = transforms.ToTensor()

    # Load training datasets
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
    ]
    mnist_train = datasets.MNIST(
    "./data", train=True, download=True, transform=transform
    )
    #mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    # Federated: legd loaders for each client
    mnist_loader = torch.utils.data.DataLoader(mnist_train, batch_size=cfg.batch_size, shuffle=True)
    fashion_loader = torch.utils.data.DataLoader(fashion_train, batch_size=cfg.batch_size, shuffle=True)

    # Test datasets (for test loss computation)
    mnist_test = datasets.MNIST(
    "./data", train=False, download=True, transform=transform
    )
    fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=cfg.eval_batch_size, shuffle=False)
    fashion_test_loader = torch.utils.data.DataLoader(fashion_test, batch_size=cfg.eval_batch_size, shuffle=False)

    return mnist_loader, mnist_test_loader, fashion_loader, fashion_test_loader