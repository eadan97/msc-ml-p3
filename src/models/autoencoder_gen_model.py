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


class AutoencoderGenModel(LightningModule):
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
            ngf: int = 64,
            lr: float = 1e-4,
            weight_decay: float = 0.0005,
            no_dropout: bool = True,
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

        super(AutoencoderGenModel, self).__init__()

        self.save_hyperparameters()
        self.lambda_style = lambda_style
        self.lambda_features = lambda_features
        self.lambda_tv = lambda_tv
        self.ngf = ngf

        self.encoder = ResnetEncoder(3, 3, self.ngf,
                                     norm_layer=functools.partial(nn.InstanceNorm2d, affine=False,
                                                                  track_running_stats=False),
                                     use_dropout=not no_dropout, n_blocks=9)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 1, 1)),
            nn.ConvTranspose2d(256, 512, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False),#1
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 1024, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False),#1
            nn.InstanceNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),#2
            nn.InstanceNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),#4
            nn.InstanceNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False),#7
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),#14
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),#28
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),#56
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            ResnetDecoder(3, 3, 64,
                          norm_layer=functools.partial(nn.InstanceNorm2d, affine=False,
                                                       track_running_stats=False),
                          use_dropout=not no_dropout, n_blocks=9))

        self.fc = nn.Sequential(

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # sale 7
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.ReLU(),
            ResnetMiddle(3, 3, 256,
                         norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
                         use_dropout=not no_dropout, n_blocks=9, padding_type='zero'),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )

        # self.encoder.to(device=self.device)
        # self.decoder.to(device=self.device)
        # self.fc.to(device=self.device)
        # loss function
        self.perce_criterion = PerceptionLoss()
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
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

        self.perce_criterion.set_source_image(x_hat)
        self.perce_criterion.set_target_image(x)
        # We are using only features since we are not doing style transfer
        loss = self.lambda_features * self.perce_criterion.get_feature_loss()

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
