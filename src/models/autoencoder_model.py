from typing import Any, List, Optional

import torch
from pl_bolts.models import AE
from pl_bolts.models.autoencoders.components import ResNetEncoder, ResNetDecoder, DecoderBottleneck, EncoderBottleneck, \
    EncoderBlock, DecoderBlock
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn
from torchmetrics.classification.accuracy import Accuracy
from torchvision.models import resnext101_32x8d

from src.callbacks.tensorboardx_callbacks import get_tensorboard_logger
from src.losses.perception_loss import PerceptionLoss
from src.models.modules.simple_dense_net import SimpleDenseNet


class AutoencoderModel(AE):
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
            latent_dim: int = 256,
            lr: float = 1e-4,
            enc_type: str = 'resnet18',
            enc_out_dim: int = 512,
            lambda_features: float = 1.0,
            lambda_style: float = 1.0,
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

        super(AutoencoderModel, self).__init__(input_height, first_conv=first_conv, maxpool1=maxpool1,
                                               latent_dim=latent_dim, lr=lr, enc_type=enc_type, enc_out_dim=enc_out_dim)

        self.save_hyperparameters()
        self.perce_criterion = PerceptionLoss()

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        return z

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = 0.0
        if self.hparams.lambda_features > 0 or self.hparams.lambda_style > 0:
            self.perce_criterion.set_source_image(x_hat)
            self.perce_criterion.set_target_image(x)
        if self.hparams.lambda_features > 0:
            loss += self.hparams.lambda_features * self.perce_criterion.get_feature_loss()
        if self.hparams.lambda_style > 0:
            loss += self.hparams.lambda_style * self.perce_criterion.get_style_loss()
        return loss, {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        return {'latent_vector': z.detach(), 'generated_image': x_hat.detach(), 'label': y.detach()}

    def test_epoch_end(self, outputs):
        logger = get_tensorboard_logger(trainer=self.trainer)
        experiment = logger.experiment

        test_labels = torch.cat([item['label'] for item in outputs])[:500]
        test_imgs = torch.cat([item['generated_image'] for item in outputs])[:500]
        test_vectors = torch.cat([item['latent_vector'] for item in outputs])[:500]
        string_labels = [self.trainer.datamodule.classes[i] for i in test_labels]
        # run the batch through the network

        experiment.add_images("generated_images", test_imgs)
        experiment.add_embedding(test_vectors, string_labels, test_imgs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

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
