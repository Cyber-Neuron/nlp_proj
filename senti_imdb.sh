bas=`pwd`
DATA_DIR=$bas'/data/'
TMP_DIR=$bas'/tmp/'
OUT_DIR=$bas/$1
t2t-trainer \
  --model=transfer_transformer \
  --hparams_set=transformer_small \
  --train_steps=$3 \
  --eval_steps=400 \
  --data_dir=$DATA_DIR \
  --t2t_usr_dir=$2 \
  --hparams='shared_embedding=True,batch_size=64,learning_rate=0.00025,learning_rate_decay_steps=2000' \
  --output_dir=$OUT_DIR \
  --local_eval_frequency=100 \
  --eval_throttle_seconds=10 \
  --problem=$1 \
  --worker_gpu=$4 \
  --warm_start_from=$5 #path of pre-trained model
