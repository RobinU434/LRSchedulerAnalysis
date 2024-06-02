"""This code was adapted from: 
https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
"""

from typing import Any, Dict, Type

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from lightning.pytorch import LightningModule


class CIFARLitModule(LightningModule):
    def __init__(
        self,
        model_cls: Type[nn.Module],
        model_hparams: Dict[str, Any] = dict(),
        optimizer_name: str = "SGD",
        optimizer_hparams: Dict[str, Any] = dict(),
        lr_scheduler_name: str = "ConstantLR",
        lr_scheduler_hparams: Dict[str, Any] = dict(),
    ):
        """CIFARModule.

        Args:
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model_cls(**model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # AdamW is Adam with a correct implementation of weight decay (see here
        # for details: https://arxiv.org/pdf/1711.05101.pdf)
        try:
            optimizer_cls = getattr(optim, self.hparams.optimizer_name)
            optimizer = optimizer_cls(
                self.parameters(), **self.hparams.optimizer_hparams
            )
        except AttributeError:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        try:
            scheduler_cls = getattr(lr_scheduler, self.hparams.lr_scheduler_name)
            if self.hparams.lr_scheduler_name == "SequentialLR":
                scheduler1 = lr_scheduler.LinearLR(
                    optimizer=optimizer,
                    start_factor=self.hparams.lr_scheduler_hparams["start_factor"],
                    total_iters=self.hparams.lr_scheduler_hparams["total_iters"],
                )
                # scheduler2 = lr_scheduler.ConstantLR(optimizer=optimizer, factor=1)
                scheduler2 = lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.hparams.lr_scheduler_hparams["T_max"],
                    eta_min=self.hparams.lr_scheduler_hparams["eta_min"],
                )
                kwargs = {
                    "schedulers": [scheduler1, scheduler2],
                    "milestones": [self.hparams.lr_scheduler_hparams["total_iters"]],
                }
            else:
                kwargs = self.hparams.lr_scheduler_hparams

            scheduler = scheduler_cls(optimizer=optimizer, **kwargs)
        except AttributeError:
            assert False, f'Unknown lr_scheduler: "{self.hparams.lr_scheduler_name}"'

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        # print(imgs.shape)
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)
