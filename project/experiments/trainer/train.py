import os
import sys
from typing import Any, Dict

import torch
from experiments import CHECKPOINT_PATH
from experiments.dataset.pl_data_module import CIFARLitDataModule
from experiments.model.pl_module import CIFARLitModule
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger


def train_model(
    save_name: str = None,
    max_epochs: int = 200,
    seed: int = 42,
    model_kwargs: Dict[str, Any] = dict(),
):
    """Train model.

    Args:
        save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = "BaselineClassifier"
    seed_everything(seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create a PyTorch Lightning trainer with the generation callback
    log_path = os.path.join(CHECKPOINT_PATH, save_name)
    datamodule = CIFARLitDataModule()

    trainer = Trainer(
        default_root_dir=log_path,  # Where to save models
        # We run on a single GPU (if possible)
        accelerator="auto",
        devices=1,
        # How many epochs to train for if no patience is set
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc_epoch"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],  # Log learning rate every epoch
        logger=[CSVLogger(log_path), TensorBoardLogger(log_path)],
        check_val_every_n_epoch=5
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = (
        True  # If True, we plot the computation graph in tensorboard
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )
    model = CIFARLitModule(**model_kwargs)
    trainer.fit(model, datamodule=datamodule)

    model = CIFARLitModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )  # Load best checkpoint after training
    # Test best model on validation and test set
    val_result = trainer.test(model, datamodule=datamodule, verbose=False)
    test_result = trainer.test(model, datamodule=datamodule, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result
