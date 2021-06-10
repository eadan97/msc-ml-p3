from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.classification.accuracy import Accuracy

from src.models.modules.simple_dense_net import SimpleDenseNet


class AutoencoderModel(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            input_height: int,
            first_conv: bool = False,
            maxpool1: bool = False,
            enc_out_dim: int = 512,
            latent_dim: int = 256,
            lr: float = 1e-4,
            weight_decay: float = 0.0005,
            **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AutoencoderModel, self).__init__()

        self.save_hyperparameters()

        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1)
        self.decoder = valid_encoders[enc_type]['dec'](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)
        # loss function
        self.criterion = torch.nn.MSELoss(reduction='mean')
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = self.criterion(x_hat, x)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch, batch_idx)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #
    #     parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
    #     parser.add_argument("--first_conv", action='store_true')
    #     parser.add_argument("--maxpool1", action='store_true')
    #     parser.add_argument("--lr", type=float, default=1e-4)
    #
    #     parser.add_argument(
    #         "--enc_out_dim",
    #         type=int,
    #         default=512,
    #         help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
    #     )
    #     parser.add_argument("--latent_dim", type=int, default=256)
    #
    #     parser.add_argument("--batch_size", type=int, default=256)
    #     parser.add_argument("--num_workers", type=int, default=8)
    #     parser.add_argument("--data_dir", type=str, default=".")
    #
    #     return parser

