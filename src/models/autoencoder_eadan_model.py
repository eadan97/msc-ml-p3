import functools
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.classification.accuracy import Accuracy

from src.callbacks.tensorboardx_callbacks import get_tensorboard_logger
from src.losses.perception_loss import PerceptionLoss
from src.models.modules.resnet_generator import ResnetGenerator, ResnetEncoder, ResnetDecoder, ResnetMiddle
from src.models.modules.simple_dense_net import SimpleDenseNet
from src.models.modules.unet import DoubleConv, Down, Up


class AutoencoderEadanModel(LightningModule):
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
            input_height: int = 224,
            ngf: int = 64,
            lr: float = 1e-4,
            weight_decay: float = 0.0005,
            layers: int = 6,
            latent_dim: int = 256,
            lambda_style: float = 1.0,
            lambda_features: float = 1.0,
            lambda_tv: float = 0.000001,
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

        super(AutoencoderEadanModel, self).__init__()

        self.save_hyperparameters()
        self.lambda_style = lambda_style
        self.lambda_features = lambda_features
        self.lambda_tv = lambda_tv
        self.ngf = ngf
        self.last_conv_size = (input_height // 2 ** (layers + 1))
        encoder_layers: list = [nn.Conv2d(3, ngf, kernel_size=4, stride=2, padding=1, bias=True)]
        features = ngf
        for _ in range(layers - 1):
            encoder_layers = encoder_layers + [
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(features * 2)
            ]
            features *= 2
        encoder_layers = encoder_layers + [nn.LeakyReLU(0.2, True),
                                           nn.Conv2d(features, features, kernel_size=4, stride=2, padding=1,
                                                     bias=True)]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features * self.last_conv_size ** 2, latent_dim),
            nn.ReLU(True)
        )
        decoder_layers: list = [
            nn.Linear(latent_dim, features * self.last_conv_size ** 2),
            nn.Unflatten(1, (features, self.last_conv_size, self.last_conv_size)),
            nn.ReLU(True),
            nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=1,
                               bias=True),
            nn.InstanceNorm2d(3)]
        for _ in range(layers - 1):
            decoder_layers = decoder_layers + [
                nn.ReLU(True),
                nn.ConvTranspose2d(features, features // 2, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(3)
            ]
            features //= 2
        decoder_layers = decoder_layers + [nn.ReLU(True),
                                           nn.ConvTranspose2d(features, 3, kernel_size=4, stride=2, padding=1),
                                           nn.Sigmoid()]

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # loss function
        self.perce_criterion = PerceptionLoss()
        self.mse_criterion = nn.MSELoss()
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # self.train_accuracy = Accuracy()
        # self.val_accuracy = Accuracy()
        # self.test_accuracy = Accuracy()
        # print(dict(self.named_modules()))

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

        if self.lambda_features > 0 or self.lambda_style > 0:
            self.perce_criterion.set_source_image(x_hat)
            self.perce_criterion.set_target_image(x)
        if self.lambda_features > 0:
            loss += self.lambda_features * self.perce_criterion.get_feature_loss()
        if self.lambda_style > 0:
            loss += self.lambda_style * self.perce_criterion.get_style_loss()
        if self.lambda_tv > 0:
            loss += self.lambda_tv * self.mse_criterion(x, x_hat)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # log train metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        return {'latent_vector': z.detach().cpu(), 'generated_image': x_hat.detach().cpu(), 'label': y.detach().cpu()}

    def test_epoch_end(self, outputs):
        logger = get_tensorboard_logger(trainer=self.trainer)
        experiment = logger.experiment

        test_labels = torch.cat([item['label'] for item in outputs[:50]])
        test_imgs = torch.cat([item['generated_image'] for item in outputs[:50]])
        test_vectors = torch.cat([item['latent_vector'] for item in outputs[:50]])
        test_vectors = torch.flatten(test_vectors, start_dim=1)
        string_labels = [self.trainer.datamodule.classes[i] for i in test_labels]
        # run the batch through the network
        print(test_vectors.shape, test_imgs.shape, test_labels.shape)

        experiment.add_images("generated_images", test_imgs)
        experiment.add_embedding(test_vectors, string_labels, test_imgs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
