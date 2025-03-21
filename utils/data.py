import torch
from torchvision import datasets, transforms

__all__ = ['get_dataloaders', 'idx2onehot']

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
    if cfg.model.conditional:
        fashion_train.targets = fashion_train.targets + 10

    # Federated: legd loaders for each client
    mnist_loader = torch.utils.data.DataLoader(mnist_train, batch_size=cfg.batch_size, shuffle=True)
    fashion_loader = torch.utils.data.DataLoader(fashion_train, batch_size=cfg.batch_size, shuffle=True)

    # Test datasets (for test loss computation)
    mnist_test = datasets.MNIST(
    "./data", train=False, download=False, transform=transform
    )
    fashion_test = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    if cfg.model.conditional:
        fashion_test.targets = fashion_test.targets + 10
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=cfg.eval_batch_size, shuffle=False)
    fashion_test_loader = torch.utils.data.DataLoader(fashion_test, batch_size=cfg.eval_batch_size, shuffle=False)

    return mnist_loader, mnist_test_loader, fashion_loader, fashion_test_loader



def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot
