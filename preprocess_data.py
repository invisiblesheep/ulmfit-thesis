from argparse import ArgumentParser

from fastai import *
from fastai.text import *
from os import path


parser = ArgumentParser()
parser.add_argument("--folder", help="source folder of text data")
parser.add_argument("--sp_model", help="location of sentencepiece model")
parser.add_argument("--sp_vocab", help="location of sentencepiece vocab")
parser.add_argument("--vocab_size", help="vocab size", default=30000)
parser.add_argument("--p", help="percentage of data to use", type=int, default=100)
parser.add_argument("--split", help="percentage of data to use as validation", default=0.33)
parser.add_argument("--nrows", help="number of rows to read", type=int)


ARGS = parser.parse_args()

FOLDER = ARGS.folder
PERCENTAGE = ARGS.p
NROWS = ARGS.nrows
SP_MODEL = ARGS.sp_model
SP_VOCAB = ARGS.sp_vocab
VOCAB_SIZE = ARGS.vocab_size

if SP_MODEL == None:
    #SP_MODEL = Path(FOLDER) / 'wsl.model'
    SP_MODEL = f'../vocabs/wsl.{VOCAB_SIZE}.model'

if SP_VOCAB == None:
    #SP_VOCAB = Path(FOLDER) / 'wsl.vocab'
    SP_VOCAB = f'../vocabs/ulmfit.{VOCAB_SIZE}.vocab'

SPLIT = ARGS.split
DELIMITER = "\t"
HEADER = None
print(f"percentage: {PERCENTAGE}, nrows: {NROWS}")
print(f'sp_model: {SP_MODEL}, sp_vocab: {SP_VOCAB}')


def preprocess():
    """Run preprocessing for data in csv
    """

    if path.exists(Path(FOLDER) / 'processed/vocab_{VOCAB_SIZE}/train_ids.npy'):
        print('Data already processed!')

        return

    if NROWS is not None:
        df = pd.read_csv(
            Path(FOLDER) / 'all_texts.csv', delimiter=DELIMITER, header=HEADER, nrows=NROWS
        )
    else:
        df = pd.read_csv(Path(FOLDER) / 'all_texts.csv', delimiter=DELIMITER, header=HEADER)

    df = df.dropna()
    df = df.iloc[np.random.permutation(len(df))]
    cut = int(PERCENTAGE * len(df)) + 1
    df = df[:cut]

    print("Data loaded! Starting preprocessing")

    spprocessor = SPProcessor(
        sp_model=SP_MODEL,
        sp_vocab=SP_VOCAB
    )

    data = TextList.from_df(df, path=FOLDER, processor=spprocessor)
    #data = data.split_by_rand_pct(SPLIT)
    data = data.split_by_rand_pct()
    print("Data split")
    data = data.label_for_lm()
    print("Data labelled")
    if not os.path.exists(Path(FOLDER) / 'processed'):
        os.mkdir(Path(FOLDER) / 'processed')

    if not os.path.exists(Path(FOLDER) / f'processed'/ f'vocab_{VOCAB_SIZE}'):
        os.mkdir(Path(FOLDER) / 'processed'/ f'vocab_{VOCAB_SIZE}')

    np.save(Path(FOLDER) / f'processed/vocab_{VOCAB_SIZE}/train_ids.npy', data.train.x.items)
    np.save(Path(FOLDER) / f'processed/vocab_{VOCAB_SIZE}/train_labels.npy', data.train.y.items)
    np.save(Path(FOLDER) / f'processed/vocab_{VOCAB_SIZE}/val_ids.npy', data.valid.x.items)
    np.save(Path(FOLDER) / f'processed/vocab_{VOCAB_SIZE}/val_labels.npy', data.valid.y.items)
    data.train.vocab.save(Path(FOLDER) / f'processed/vocab_{VOCAB_SIZE}/itos.pkl')
    print(f"Data saved in {FOLDER}/processed/")


preprocess()
