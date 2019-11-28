from sentencepiece import SentencePieceTrainer
import fire

def train_sentencepiece(sentence_file, sentencepiece_model_name, vocab_size):
  SentencePieceTrainer.Train(
      " ".join(
          [
              f"--input={sentence_file}",
              f"--character_coverage=1.0",
              f"--unk_id=0 --pad_id=-1 --bos_id=-1 --eos_id=-1",
              f"--input_sentence_size=2000000 --shuffle_input_sentence=true",
              f"--model_prefix={sentencepiece_model_name} --vocab_size={vocab_size} --model_type=unigram",
          ]
      )
  )

if __name__ == "__main__":
    fire.Fire(train_sentencepiece)
