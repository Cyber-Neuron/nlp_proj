bas=`pwd`
DATA_DIR=$bas'/data/'
TMP_DIR=$bas'/tmp/'
OUT_DIR=$bas'/output/'
t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_base \
  --train_steps=1000 \
  --eval_steps=8 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --t2t_usr_dir=$1 \
  --problem=$2

