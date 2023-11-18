MODEL_GROUP=$1
MODEL_NAME=$(basename $MODEL_GROUP)

echo "MODEL_GROUP=$MODEL_GROUP"
echo "MODEL_NAME=$MODEL_NAME"

python generate_eval_jobs.py $MODEL_GROUP /home/project/13003565/geyu/jobs/seaeval_job/$MODEL_NAME hidden_test 