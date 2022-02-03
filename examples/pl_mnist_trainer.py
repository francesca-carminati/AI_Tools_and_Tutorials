import neptune.new as neptune
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from decouple import config
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# TODO: change params and add neptune logger

BATCH_SIZE = config("BATCH_SIZE", cast=int, default=1)
MODEL_DIR = config("MODEL_DIR", default="/workspace/model")
DATA_DIR = config("DATA_DIR", default="/workspace/data")

# Neptune API
NEPTUNE_API_TOKEN = config("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = config("NEPTUNE_PROJECT")

# Define parameters
MODEL_PARAM = {"image_size": 28, "linear": 128, "n_classes": 10, "learning_rate": 0.0023, "decay_factor": 0.95}

DATA_PARAM = {
    "batch_size": BATCH_SIZE,
    "num_workers": 4,
    "normalization_vector": ((0.1307,), (0.3081,)),
}

LR_PARAM = {"logging_interval": "epoch"}

CHECKPOINT_PARAM = {
    "filename": "{MODEL_DIR}/checkpoints/{epoch:02d}-{val_loss:.2f}",
    "save_weights_only": True,
    "save_top_k": 3,
    "monitor": "val_loss",
}

TRAINER_PARAM = {"log_every_n_steps": 100, "max_epochs": 5, "gpus": 1, "track_grad_norm": 2}

ALL_PARAMS = {**MODEL_PARAM, **DATA_PARAM, **LR_PARAM, **CHECKPOINT_PARAM, **TRAINER_PARAM}

run = neptune.init(
    api_token=NEPTUNE_API_TOKEN,
    project=NEPTUNE_PROJECT,
    name="pl-with-active-learning",
    description="Detailed example of how to train a model using pytorch lightning",
    tags=["PL", "image", "MNIST_dataset", "detailed-example"],
)

run["hyper-parameters"] = ALL_PARAMS
neptune_logger = NeptuneLogger(run=run)


class LitModel(pl.LightningModule):
    def __init__(self, image_size, linear, n_classes, learning_rate, decay_factor):
        super().__init__()
        self.image_size = image_size
        self.linear = linear
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.train_img_max = 10
        self.train_img = 0

        self.layer_1 = torch.nn.Linear(image_size * image_size, linear)
        self.layer_2 = torch.nn.Linear(linear, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, lambda epoch: self.decay_factor ** epoch)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("metrics/batch/loss", loss, prog_bar=False)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/batch/acc", acc)

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("metrics/batch/val_loss", loss, prog_bar=False)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("metrics.batch/test_loss", loss, prog_bar=False)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def training_epoch_end(self, outputs):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/epoch/loss", loss.mean())
        self.log("metrics/epoch/acc", acc)

    def validation_epoch_end(self, outputs):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/epoch/val_loss", loss.mean())
        self.log("metrics/epoch/val_acc", acc)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, normalization_vector):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalization_vector = normalization_vector

    def prepare_data(self):
        MNIST(DATA_DIR, train=True, download=True)
        MNIST(DATA_DIR, train=False, download=True)

    def setup(self, stage):
        # transforms
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.normalization_vector[0], self.normalization_vector[1])]
        )

        if stage == "fit":
            mnist_train = MNIST(DATA_DIR, download=True, train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == "test":
            self.mnist_test = MNIST(DATA_DIR, download=True, train=False, transform=transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
        return mnist_test


if __name__ == "__main__":
    seed_everything(42)
    lr_logger = LearningRateMonitor(**LR_PARAM)
    model_checkpoint = ModelCheckpoint(**CHECKPOINT_PARAM)

    trainer = pl.Trainer(
        logger=neptune_logger, checkpoint_callback=model_checkpoint, callbacks=[lr_logger], **TRAINER_PARAM
    )

    model = LitModel(**MODEL_PARAM)
    dm = MNISTDataModule(**DATA_PARAM)

    trainer.fit(model, dm)
    trainer.test(datamodule=dm, ckpt_path=trainer.checkpoint_callback.last_model_path)
    model.freeze()

    neptune_logger.log_model_summary(model=model, max_depth=-1)
    trainer.save_checkpoint(f"{MODEL_DIR}/image_MNIST_classifier.pt")
    run.stop()
