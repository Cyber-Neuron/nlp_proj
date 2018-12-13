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
  --t2t_usr_dir=$2 \
  --hparams='shared_embedding=True,num_decoder_layers=12,num_heads=12,hidden_size=768' \
  --output_dir=$OUT_DIR \
  --local_eval_frequency=1000 \
  --eval_throttle_seconds=10 \
  --problem=$1 \
  --warm_start_from=$4 #path of pre-trained model
