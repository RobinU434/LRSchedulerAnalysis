from project.experiments.dataset.utils import download_cifar_10


class Experiments:
    def __init__(self) -> None:
        pass

    def start_training(self):
        pass


    def download_dataset(self, target: str):
        """download datasets needed for the experiments

        Currently downloaded:
        - CIFAR10

        Args:
            target (str): where to store the dataset
        """

        download_cifar_10(target)