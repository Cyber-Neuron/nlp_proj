bas=`pwd`
DATA_DIR=$bas'/data/'
TMP_DIR=$bas'/tmp/'
OUT_DIR=$bas'/output/'
OUT_DIR2=$bas'/output2/'
t2t-trainer \
  --model=transfer_transformer \
  --hparams_set=transformer_small \
  --train_steps=4000 \
  --eval_steps=5 \
  --data_dir=$DATA_DIR \
  --t2t_usr_dir=$1 \
  --hparams='shared_embedding=True' \
  --output_dir=$OUT_DIR \
  --problem=amzcls 
#--warm_start_from=$OUT_DIR2 \
