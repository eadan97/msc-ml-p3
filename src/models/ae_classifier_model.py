from math import ceil
from typing import Any, List

import pandas as pd
import seaborn
import torch
import torchmetrics
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.classification.accuracy import Accuracy

from src.callbacks.tensorboardx_callbacks import get_tensorboard_logger
from src.models.autoencoder_eadan_model import AutoencoderEadanModel
from src.models.modules.simple_dense_net import SimpleDenseNet
from src.utils import utils


class AutoencoderClassifierModel(LightningModule):

    def __init__(
            self,
            checkpoint_dir: str,
            freeze_autoencoder: bool = True,
            latent_dim: int = 2048,
            num_neurons: int = 0,
            lr: float = 0.001,
            weight_decay: float = 0.0,
            **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.autoencoder = AutoencoderEadanModel.load_from_checkpoint(checkpoint_dir)

        if freeze_autoencoder:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        else:
            for param in self.autoencoder.decoder.parameters():
                param.requires_grad = False

        num_classes = 39
        if num_neurons == 0:
            num_neurons = 8 * ceil(num_classes / 8)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, num_neurons * 8),
            nn.BatchNorm1d(num_neurons * 8),
            nn.ReLU(),
            nn.Linear(num_neurons * 8, num_neurons * 4),
            nn.BatchNorm1d(num_neurons * 4),
            nn.ReLU(),
            nn.Linear(num_neurons * 4, num_neurons * 2),
            nn.BatchNorm1d(num_neurons * 2),
            nn.ReLU(),
            nn.Linear(num_neurons * 2, num_classes),
            # nn.Softmax()
        )

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.class_accuracy = torchmetrics.Accuracy(average='none', num_classes=num_classes)
        # self.auroc = torchmetrics.AUROC(num_classes=num_classes, average="weighted")

    def forward(self, x: torch.Tensor):
        latent_vector = self.autoencoder.encoder(x)
        latent_vector = self.autoencoder.fc(latent_vector)
        logits = self.model(latent_vector)
        return logits

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)
        preds = torch.argmax(logits, dim=1)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "logits": logits, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)
        preds = torch.argmax(logits, dim=1)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "logits": logits, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)
        preds = torch.argmax(logits, dim=1)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "logits": logits, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        file_log = utils.get_logger(__name__)

        logger = get_tensorboard_logger(trainer=self.trainer)
        experiment = logger.experiment

        logits = torch.cat([tmp['logits'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        # prob = nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        # aucroc = self.auroc(prob, targets)
        class_acc = self.class_accuracy(preds, targets)

        # Calculate, plot and log the class accuracy as bar plot
        acc_data = [[label, val.item()] for (label, val) in zip(self.trainer.datamodule.classes, class_acc)]
        acc_dict = {}
        for label, val in acc_data:
            acc_dict[label] = val
        file_log.info(str(acc_dict))
        # experiment.add_scalars('class_accuracy', acc_dict)
        acc_table = pd.DataFrame(acc_data, columns=["class", "accuracy"])
        acc_table = acc_table.explode("accuracy")
        plt.figure(figsize=(30, 21))
        barplot = seaborn.barplot(x="accuracy", y="class", data=acc_table).get_figure()
        plt.close(barplot)
        experiment.add_figure("Class_Accuracy", barplot)

        # Calculate, plot and log the confusion matrix
        confusion_matrix = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=39)
        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index=self.trainer.datamodule.classes,
                             columns=self.trainer.datamodule.classes)
        plt.figure(figsize=(30, 21))
        seaborn.set(font_scale=1.4)
        fig_ = seaborn.heatmap(df_cm, annot=True, cmap='Purples').get_figure()
        plt.close(fig_)
        experiment.add_figure("Confusion_Matrix", fig_)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
