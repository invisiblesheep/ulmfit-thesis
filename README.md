# ULMFit + SentencePiece

A pipeline for training ULMFit with SentencePiece encoding.
Uses fastai.

# Usage

    pip install -r requirements.txt
    ./run_pipeline.sh -d '/path/to/data/folder' -f 'file.txt' -b 128 -e 20 -v 30000

-b for batch size, -e for number of epocs, -v for vocabulary size
-d points to the folder in which the single text file resides
-f for the name of the datafile
Optionally -m for pretrained model source (to continue training) and -l for learning rate

The scripts can also be used separately.

