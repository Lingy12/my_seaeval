

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


#####
MODEL_PATH=../llm_train/output_dir/model_archive/training_v0/alpaca_zh
checkpoint='checkpoint-1200'
MODEL_NAME=$(basename $MODEL_PATH)
GPU=0,1
BZ=8
#####

cd $MODEL_PATH/$checkpoint
if [[ -e "./pytorch_model.bin" ]]; then
  echo "Model bin existss."
else
  echo "Model bin not exists" 
  python zero_to_fp32.py . pytorch_model.bin 
  cp ../../../../../helper_configs/* . 
  echo "Created a normal checkpoints"
fi
cp ../../../../../helper_configs/* . 
cd -
echo $CWD

EVAL_MODE=public_test
EVAL_MODE=hidden_test

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

mkdir -p log/$EVAL_MODE/$MODEL_NAME/$checkpoint


for ((i=1; i<=1; i++))
do

    bash scripts/eval_checkpoint.sh cross_mmlu $MODEL_PATH/$checkpoint $GPU $BZ $i $EVAL_MODE             2>&1 | tee log/$EVAL_MODE/$MODEL_NAME/$checkpoint/cross_mmlu_p$i.log

done




for ((i=1; i<=1; i++))
do

    bash scripts/eval_checkpoint.sh cross_logiqa $MODEL_PATH/$checkpoint $GPU $BZ $i $EVAL_MODE           2>&1 | tee log/$EVAL_MODE/$MODEL_NAME/$checkpoint/cross_logiqa_p$i.log

done
