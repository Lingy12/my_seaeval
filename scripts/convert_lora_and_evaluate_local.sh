#!/bin/bash
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

echo "MODEL_INDEX=$MODEL_INDEX"
echo "LISTEN_FOLDER=$LISTEN_FOLDER"
echo "BASE_MODEL=$BASE_MODEL"
set -e

if [[ -e "$LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32/pytorch_model*.bin" ]]; then
  echo "Model bin exists."
else
  echo "Model bin not exists" 
  mkdir -p $LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32
  # cp -r ./helper_configs_gemma/* $LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32
  cp $BASE_MODEL/tokenizer* $LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32
  cp $BASE_MODEL/config.json $BASE_MODEL/generation_config.json $BASE_MODEL/special_tokens_map.json $LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32
  python convert_lora.py $LISTEN_FOLDER/$MODEL_INDEX $BASE_MODEL $LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32/
fi

# mkdir -p converted_checkpoint/$MODEL_INDEX-fp32
# cp -r converted_checkpoint/helper_configs/* converted_checkpoint/$MODEL_INDEX-fp32/
# python zero_to_fp32.py $LISTEN_FOLDER/$MODEL_INDEX $LISTEN_FOLDER/$MODEL_INDEX-fp32/pytorch_model.bin

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

#mkdir -p /data/projects/13003558/pretrain_output_results/$MODEL_INDEX-results

MODEL_PATH=$LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32
echo $MODEL_PATH
##### 
MODEL_NAME=$MODEL_PATH
GPU=$CUDA_VISIBLE_DEVICES
BZ=16
#####

# for EVAL_MODE in public_test_few_shot hidden_test_few_shot
# do

#     mkdir -p converted_checkpoint/$MODEL_NAME-results/$EVAL_MODE

#     for ((i=1; i<=1; i++))
#     do
#         bash eval.sh cross_mmlu $MODEL_NAME $GPU $BZ $i $EVAL_MODE              2>&1 | tee converted_checkpoint/$MODEL_NAME-results/$EVAL_MODE/cross_mmlu_p$i.log
#         bash eval.sh cross_logiqa $MODEL_NAME $GPU $BZ $i $EVAL_MODE            2>&1 | tee converted_checkpoint/$MODEL_NAME-results/$EVAL_MODE/cross_logiqa_p$i.log
#     done
# done


echo "MODEL_NAME=$MODEL_NAME"

for EVAL_MODE in hidden_test
do
    TARGET_DIR=$LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-results/$EVAL_MODE
    mkdir -p $TARGET_DIR/log
    for ((i=1; i<=1; i++))
    do
	bash scripts/eval.sh cross_xquad $MODEL_NAME $GPU $BZ $i $EVAL_MODE $TARGET_DIR        2>&1 | tee $TARGET_DIR/log/cross_xquad_p$i.log
        bash scripts/eval.sh cross_mmlu $MODEL_NAME $GPU $BZ $i $EVAL_MODE $TARGET_DIR         2>&1 | tee $TARGET_DIR/log/cross_mmlu_p$i.log
        bash scripts/eval.sh cross_logiqa $MODEL_NAME $GPU $BZ $i $EVAL_MODE $TARGET_DIR           2>&1 | tee $TARGET_DIR/log/cross_logiqa_p$i.log
    done
done

# rm -rf $MODEL_PATH
#rm -rf $LISTEN_FOLDER/$MODEL_INDEX
# echo "$MODEL_PATH CLEANED"




#cp -r converted_checkpoint/$MODEL_INDEX-results/* /data/projects/13003558/pretrain_output_results/$MODEL_INDEX-results/

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =




