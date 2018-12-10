bas=`pwd`
DATA_DIR=$bas'/data/'
TMP_DIR=$bas'/tmp/'
OUT_DIR=$bas/$1
t2t-trainer \
  --model=transfer_transformer \
  --hparams_set=transformer_small \
  --train_steps=$3 \
  --eval_steps=1 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --t2t_usr_dir=$2 \
  --hparams='batch_size=1024,shared_embedding=True'\
  --problem=$1
  #--dataidx=3 #0:test, 1:books, 2:movies, 3:electronics

