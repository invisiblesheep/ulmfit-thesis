from argparse import ArgumentParser

from fastai import *
from fastai.text import *
from os import path


parser = ArgumentParser()
parser.add_argument("--folder", help="source folder of text data")
parser.add_argument("--filename", help="name of train file")
parser.add_argument("--sp_model", help="location of sentencepiece model")
parser.add_argument("--sp_vocab", help="location of sentencepiece vocab")
parser.add_argument("--p", help="percentage of data to use", type=int, default=100)
parser.add_argument("--nrows", help="number of rows to read", type=int)


ARGS = parser.parse_args()

FOLDER = ARGS.folder
FILENAME = ARGS.filename
PERCENTAGE = ARGS.p
NROWS = ARGS.nrows
SP_MODEL = ARGS.sp_model
SP_VOCAB = ARGS.sp_vocab
DELIMITER = "\t"
HEADER = None
print(f"percentage: {PERCENTAGE}, nrows: {NROWS}")


def preprocess():
    """Run preprocessing for data in csv
    """

    if path.exists(Path(FOLDER) / 'processed/train_ids.npy'):
        print('Data already processed!')

        return

    if NROWS is not None:
        df = pd.read_csv(
            Path(FOLDER) / FILENAME, delimiter=DELIMITER, header=HEADER, nrows=NROWS
        )
    else:
        df = pd.read_csv(Path(FOLDER) / FILENAME, delimiter=DELIMITER, header=HEADER)

    df = df.dropna()
    df = df.iloc[np.random.permutation(len(df))]
    cut = int(PERCENTAGE * len(df)) + 1
    df = df[:cut]

    print("Data loaded! Starting preprocessing")

    spprocessor = SPProcessor(
        lang="fi",
        sp_model=SP_MODEL,
        sp_vocab=SP_VOCAB
    )

    data = TextList.from_df(df, path=FOLDER, processor=spprocessor)
    data = data.split_by_rand_pct(0.33)
    print("Data split")
    data = data.label_for_lm()
    print("Data labelled")

    np.save(Path(FOLDER) / 'processed/train_ids.npy', data.train.x.items)
    np.save(Path(FOLDER) / 'processed/train_labels.npy', data.train.y.items)
    np.save(Path(FOLDER) / 'processed/val_ids.npy', data.valid.x.items)
    np.save(Path(FOLDER) / 'processed/val_labels.npy', data.valid.y.items)
    data.train.vocab.save(Path(FOLDER) / 'processed/itos.pkl')
    print(f"Data saved in {FOLDER}/processed/")


preprocess()
