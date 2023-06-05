from pathlib import Path

from dvclive.lightning import DVCLiveLogger
from lightning import Trainer

from datamodules import Segmentation3D_DataModule
from litmodels import Segmentation3D_TrainingLitModel
from utilities.misc import compose_config


def main():
    config = compose_config(Path('./config'))

    datamodule = Segmentation3D_DataModule(config=config)
    model = Segmentation3D_TrainingLitModel(config=config)
    logger = DVCLiveLogger()  # TODO: check logger params
    trainer = Trainer(
        logger=logger,
        **config['training']['trainer_args']
    )
    trainer.fit(model=model, datamodule=datamodule)
