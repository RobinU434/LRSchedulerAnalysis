# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
import os


DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "results/saved_models/")
