from functools import partial

import flash
import neptune.new as neptune
from decouple import config
from flash.core.classification import LogitsOutput
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier
from flash.image.classification.integrations.baal import ActiveLearningDataModule, ActiveLearningLoop
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import NeptuneLogger
from torch import nn, optim

# https://github.com/ElementAI/baal/blob/master/experiments/pytorch_lightning/lightning_flash_example.py

INIT_NUM_LABELS = 10
DATASET_NAME = "hymenoptera_data"
DATASET_URL = "https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip"

BATCH_SIZE = config("BATCH_SIZE", cast=int, default=1)
MODEL_DIR = config("MODEL_DIR", default="/workspace/model")
DATA_DIR = config("DATA_DIR", default="/workspace/data")
CACHE_DIR = config("CACHE_DIR", default="./cache/huggingface")
LOG_DIR = config("LOG_DIR", default="/workspace/logs")

# Neptune API
NEPTUNE_API_TOKEN = config("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = config("NEPTUNE_PROJECT")

run = neptune.init(
    api_token=NEPTUNE_API_TOKEN,
    project=NEPTUNE_PROJECT,
    name="pl-with-active-learning",
    description="Fine-tuning an image classifier using a simple pytorch lightning flash example",
    tags=["PL", "image", "active_learning"],
)


class DataModule_(ImageClassificationData):
    @property
    def num_classes(self):
        return 10


def get_model(dm):
    NUM_CLASSES = dm.num_classes

    loss_fn = nn.CrossEntropyLoss()
    head = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, NUM_CLASSES),
    )
    model = ImageClassifier(
        num_classes=NUM_CLASSES,
        head=head,
        backbone="vgg16",
        pretrained=True,
        loss_fn=loss_fn,
        optimizer=partial(optim.SGD, momentum=0.9, weight_decay=5e-4),
        learning_rate=0.001,
        # Note the serializer to Logits to be able to estimate uncertainty.
        output=LogitsOutput(),
    )
    return model


def get_head():
    head = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(512, INIT_NUM_LABELS),)
    return head


if __name__ == "__main__":
    seed_everything(42)
    neptune_logger = NeptuneLogger(run=run)

    download_data(DATASET_URL, DATA_DIR)

    dm = ActiveLearningDataModule(
        ImageClassificationData.from_folders(train_folder=f"{DATA_DIR}/{DATASET_NAME}/train/", batch_size=BATCH_SIZE),
        initial_num_labels=INIT_NUM_LABELS,
    )
    dm_predict = ImageClassificationData.from_files(
        predict_files=[f"{DATA_DIR}/{DATASET_NAME}/val/bees/65038344_52a45d090d.jpg"], batch_size=BATCH_SIZE,
    )

    model = get_model(dm=dm)
    # model = ImageClassifier(
    #     backbone="resnet18", head=get_head(), num_classes=dm.num_classes, output=ProbabilitiesOutput()
    # )

    trainer = flash.Trainer(logger=neptune_logger, max_epochs=50, gpus=1,)

    # (Optional) active learning loop Create the active learning loop and connect it to the trainer
    active_learning_loop = ActiveLearningLoop(label_epoch_frequency=1)
    active_learning_loop.connect(trainer.fit_loop)
    trainer.fit_loop = active_learning_loop

    trainer.finetune(model, datamodule=dm, strategy="freeze")
    predictions = trainer.predict(model, datamodule=dm)

    trainer.save_checkpoint(f"{MODEL_DIR}/image_classification_model.pt")
    run.stop()
