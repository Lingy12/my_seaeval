
# MODEL_NAME=/home/geyu/projects/multi-lang/x-LLM/model/m-llama-7b/
MODEL_NAME=$1
MODEL_BASE_NAME=$(basename $MODEL_NAME)
GPU=$2
BZ=$4
#####

echo $MODEL_BASE_NAME
EVAL_MODE=$3
#EVAL_MODE=hidden_test

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

mkdir -p log/$EVAL_MODE/$MODEL_BASE_NAME


for ((i=1; i<=1; i++))
do

    bash scripts/eval.sh cross_xquad $MODEL_NAME $GPU $BZ $i $EVAL_MODE             2>&1 | tee log/$EVAL_MODE/$MODEL_BASE_NAME/cross_xquad_p$i.log

done

