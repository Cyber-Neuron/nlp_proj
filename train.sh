bas=`pwd`
DATA_DIR=$bas'/data/'
TMP_DIR=$bas'/tmp/'
OUT_DIR=$bas'/output/'
t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_tiny \
  --problem=sentiment_imdb \
  --train_steps=1000 \
  --eval_steps=1 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR
