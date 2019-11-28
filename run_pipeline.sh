#!/usr/bin/env bash
set -e

while getopts ":d:f:b:e:l:m:v:" opt; do
  case $opt in
    d) DATA_DIR="$OPTARG"
    ;;
    f) SENTENCE_FILE="$OPTARG"
    ;;
    b) BS="$OPTARG"
    ;;
    e) EPOCHS="$OPTARG"
    ;;
    l) LR="$OPTARG"
    ;;
    m) MODEL_NAME="$OPTARG"
    ;;
    v) VOCAB_SIZE="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

printf "Argument DATA_DIR is %s\n" "$DATA_DIR"
printf "Argument SENTENCE_FILE is %s\n" "$SENTENCE_FILE"
printf "Argument BS is %s\n" "$BS"
printf "Argument EPOCHS is %s\n" "$EPOCHS"
printf "Argument LR is %s\n" "$LR"
printf "Argument MODEL_NAME is %s\n" "$MODEL_NAME"
printf "Argument VOCAB_SIZE is %s\n" "$VOCAB_SIZE"

OUTPUT_DIR="${DATA_DIR}/processed/up_low/tmp"
SENTENCEPIECE_MODEL_NAME="${OUTPUT_DIR}/sp-${VOCAB_SIZE}.model"
SENTENCEPIECE_VOCAB_NAME="${OUTPUT_DIR}/sp-${VOCAB_SIZE}.vocab"


./train_sentencepiece.sh $VOCAB_SIZE $DATA_DIR $SENTENCE_FILE 

python preprocess_data.py --folder=$DATA_DIR --filename=$SENTENCE_FILE --sp_model=$SENTENCEPIECE_MODEL_NAME --sp_vocab=$SENTENCEPIECE_VOCAB_NAME

# For distributed training (multiple gpus)
# May have to tweak nproc_per_node -variable..?
#python -m torch.distributed.launch --nproc_per_node=4 train_lstm_distr.py --batch_size=$BS --source=$DATA_DIR --epochs=$EPOCHS --pretrained_name=$MODEL_NAME --lr=$LR

# For single gpu
python train_lstm.py --batch_size=$BS --source=$DATA_DIR --epochs=$EPOCHS --pretrained_name=$MODEL_NAME --lr=$LR