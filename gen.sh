bas=`pwd`
DATA_DIR=$bas'/data/'
TMP_DIR=$bas'/tmp/'
OUT_DIR=$bas'/output/'
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --t2t_usr_dir=$1 \
  --problem=$2 
# sh grun 8 sh lm.sh amzlm_elec comp550/ 500000 8
