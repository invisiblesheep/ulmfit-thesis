# ULMFit + SentencePiece

A pipeline for training ULMFit with SentencePiece encoding.
Uses fastai.

## Usage

    pip install -r requirements.txt
    ./run_pipeline.sh -f '/path/to/data/folder' -d 'file.txt' -b 128 -e 20 -v 30000

-b for batch size, -e for number of epochs, -v for vocabulary size\
-f points to the folder in which the single text file resides, -d for the name of the datafile

Optionally -m for pretrained model source (to continue training) and -l for learning rate,\
-s for percentage of data to use for validation.

The scripts can also be used separately.

## Requirements

Requires that you have your corpus as a single text file, one sentence per line.\
Out of the box the code uses 2/3 of the data for training and 1/3 for validation.
