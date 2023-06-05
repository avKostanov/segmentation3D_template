from typing import Any, Optional

from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from datamodules.augmentations import make_augmentation_pipe
from networks import get_architecture, get_loss, get_optimizer, get_scheduler


class Segmentation3D_TrainingLitModel(LightningModule):
    def __init__(self, config: dict, debug_mode=False):
        self.config = config
        self.augmentation_pipe = make_augmentation_pipe(self.config['data']['augmentation_probability'])
        self.network = get_architecture(None, None)
        self.criterion = get_loss()
        self.debug_mode = debug_mode  # TODO: add debug mode

    def configure_optimizers(self) -> Any:
        optimizer = get_optimizer(
            self.config['training']['optimizer']['name'],
            self.config['training']['optimizer']['params']
        )

        scheduler = get_scheduler(
            optimizer=optimizer,
        )
        return {
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    def training_step(self, batch, batch_idx, dataloader_idx) -> STEP_OUTPUT:
        # TODO: figure out augmentations

        image, mask = batch['image'], batch['mask']
        predictions = self.network(image)
        loss = self.criterion(predictions, mask)

        self.log('loss/train', loss, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx) -> STEP_OUTPUT:

        image, mask = batch['image'], batch['mask']
        predictions = self.network(image)
        loss = self.criterion(predictions, mask)

        self.log('loss/val', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss


class Segmentation3D_TestingLitModel(LightningModule):
    def __init__(self, config: dict):
        self.config = config
        self.network = get_architecture(None, None)

    def test_step(self, batch, batch_idx, dataloader_idx) -> STEP_OUTPUT:
        # TODO: add activation based on num classes

        image, index = batch['image'], batch['index']
        predictions = self.network(image)

        return
