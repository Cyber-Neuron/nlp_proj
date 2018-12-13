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
  --hparams='shared_embedding=True,batch_size=64,learning_rate=0.00025,learning_rate_decay_steps=2000'\
  --local_eval_frequency=10000 \
  --worker_gpu=$4 \
  --problem=$1
  #--dataidx=3 #0:test, 1:books, 2:movies, 3:electronics
# eq: 100*8000000/(64*8)
