from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, LoggerCollection


def get_tensorboard_logger(trainer: Trainer) -> TensorBoardLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, TensorBoardLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, TensorBoardLogger):
                return logger

    raise Exception(
        "You are using tensorboard related callback, but TensorBoardLogger was not found for some reason..."
    )


class LogLatentVectorAndImageAutoencoder(Callback):
    """Logs a validation batch and their reconstruction to wandb.
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_tensorboard_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels = val_samples
            string_labels = [trainer.datamodule.classes[i] for i in val_labels]
            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            z = pl_module.encoder(val_imgs)
            latent_vector = pl_module.fc(z)
            x_hat = pl_module.decoder(latent_vector)
            experiment.add_images("val_generated_images",  x_hat, trainer.current_epoch)
