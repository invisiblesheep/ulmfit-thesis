import os
from argparse import ArgumentParser
from pathlib import Path

from fastai import *
from fastai.callbacks import CSVLogger
from fastai.callbacks.tracker import (EarlyStoppingCallback, SaveModelCallback,
                                      TrackerCallback)
from fastai.distributed import *
from fastai.text import *
from fastai.utils.mem import *

parser = ArgumentParser()


parser = ArgumentParser()
parser.add_argument(
    "--source", dest="source", help="source of train and valid files, vocab"
)
parser.add_argument("--batch_size", help="batch size", type=int)
parser.add_argument("--epochs", help="epochs", type=int)
parser.add_argument(
    "--pretrained_name", help="continue training the model from saved pth"
)
parser.add_argument("--lr", help="learning rate")
parser.add_argument("--local_rank", type=int)

args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend="nccl", init_method="env://")


BS = args.batch_size
EPOCHS = args.epochs
SOURCE = args.source
MODEL_NAME = args.pretrained_name
LR = args.lr


class SaveModelCallbackFp32(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."

    def __init__(
        self,
        learn: Learner,
        monitor: str = "valid_loss",
        mode: str = "auto",
        every: str = "improvement",
        name: str = "bestmodel",
    ):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every, self.name = every, name

        if self.every not in ["improvement", "epoch"]:
            warn(
                f'SaveModel every {self.every} is invalid, falling back to "improvement".'
            )
            self.every = "improvement"

    def jump_to_epoch(self, epoch: int) -> None:
        try:
            self.learn.load(f"{self.name}_{epoch-1}", purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except:
            print(f"Model {self.name}_{epoch-1} not found.")

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        "Compare the value monitored to its best score and maybe save the model."

        if self.every == "epoch":
            self.learn.to_fp32().save(f"{self.name}_{epoch}")
        else:  # every="improvement"
            current = self.get_monitor_value()

            if current is not None and self.operator(current, self.best):
                print(
                    f"Better model found at epoch {epoch} with {self.monitor} value: {current}."
                )
                self.best = current
                self.learn.to_fp32().save(f"{self.name}")

    def on_train_end(self, **kwargs):
        "Load the best model."

        if self.every == "improvement":
            self.learn.load(f"{self.name}", purge=False)


def train_ulmfit():
    """ Load databunch, train language model with AWD-LSTM architecture """

    trn_x = np.load(Path(SOURCE) / "processed/train_ids.npy", allow_pickle=True)
    trn_y = np.load(Path(SOURCE) / "processed/train_labels.npy", allow_pickle=True)
    val_x = np.load(Path(SOURCE) / "processed/val_ids.npy", allow_pickle=True)
    val_y = np.load(Path(SOURCE) / "processed/val_labels.npy", allow_pickle=True)
    vocab = Vocab.load(Path(SOURCE) / "processed/itos.pkl")

    data_lm = TextLMDataBunch.from_ids(
        SOURCE,
        vocab=vocab,
        train_ids=trn_x,
        valid_ids=val_x,
        train_lbls=trn_y,
        valid_lbls=val_y,
        bs=BS,
    )

    # best_model_save = partial(SaveModelCallback,
    # monitor='accuracy',
    # every='improvement',
    # name='best_lm')

    early_stop = partial(
        EarlyStoppingCallback, monitor="valid_loss", min_delta=0.01, patience=2
    )

    if MODEL_NAME is not None:
        checkpoint_save = partial(
            SaveModelCallbackFp32,
            every="epoch",
            monitor="accuracy",
            name=f"{MODEL_NAME}_continued",
        )

    else:
        checkpoint_save = partial(
            SaveModelCallbackFp32, every="epoch", monitor="accuracy", name="saved_net"
        )

    csv_logger = partial(CSVLogger, filename="models/history")

    learn = language_model_learner(
        data_lm,
        AWD_LSTM,
        drop_mult=0.1,
        wd=0.1,
        pretrained=False,
        metrics=[accuracy],
        callback_fns=[csv_logger, checkpoint_save],
    )

    if MODEL_NAME:
        print("Loading pretrained model..")
        learn.load(Path(SOURCE) / "models" / MODEL_NAME)

    # callback_fns=[early_stop, best_model_save, checkpoint_save, CSVLogger])

    if LR:
        lr = LR
    else:
        lr = 3e-3

    learn.to_fp16()
    learn = learn.to_distributed(args.local_rank)
    learn.unfreeze()
    print("Starting training..")
    learn.fit_one_cycle(EPOCHS, lr, moms=(0.8, 0.7))


train_ulmfit()
