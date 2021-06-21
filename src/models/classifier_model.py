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
from src.models.modules.simple_dense_net import SimpleDenseNet
from src.utils import utils


class ClassifierModel(LightningModule):

    def __init__(
            self,
            input_height: int = 224,
            num_neurons: int = 0,
            ngf: int = 64,
            layers: int = 5,
            lr: float = 0.001,
            weight_decay: float = 0.0,
            **kwargs
    ):
        super(ClassifierModel, self).__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
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

        self.conv_part = nn.Sequential(*encoder_layers)

        num_classes = 39
        if num_neurons == 0:
            num_neurons = 8 * ceil(num_classes / 8)

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features * self.last_conv_size ** 2, num_neurons * 8),
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

    def forward(self, x: torch.Tensor):
        out = self.conv_part(x)
        logits = self.model(out)
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
