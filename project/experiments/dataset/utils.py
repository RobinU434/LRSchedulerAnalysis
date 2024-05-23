from torchvision import datasets


def download_cifar_10(target: str):
    datasets.CIFAR10(root=target, download=True)


def download_cifar_100(target: str):
    datasets.CIFAR100(root=target, download=True)


def download_fashion_mnist(target: str):
    datasets.FashionMNIST(root=target, download=True)


def download_mnist(target: str):
    datasets.MNIST(root=target, download=True)
