#!/bin/bash
#PBS -q normal
#PBS -l select=1:ncpus=64:ngpus={{ngpu}}:mem=110gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -P 13003565
#PBS -N {{model_name_base}}
#PBS -o psb_runs/{{model_name_base}}.log


################################################# 
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: executing queue is $PBS_QUEUE
echo -e "Work folder is $PWD\n\n"

echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
#################################################
cd $PBS_O_WORKDIR
echo -e "Work folder is $PWD\n\n"

#################################################
# source /data/projects/13003565/geyu/anaconda3/etc/profile.d/conda.sh
source /data/projects/13003565/geyu/miniconda3/etc/profile.d/conda.sh 
conda activate seaeval
echo "Virtual environment activated"

#################################################
#################################################
cd {{script_dir}}

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


#####
MODEL_NAME={{model_name}}
MODEL_BASE_NAME={{model_name_base}}
GPU=0
BZ=1
#####

EVAL_MODE={{eval_mode}} # public or hidden

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

mkdir -p log/$EVAL_MODE/$MODEL_BASE_NAME


for ((i=1; i<=1; i++))
do

    bash scripts/eval.sh cross_mmlu $MODEL_NAME $GPU $BZ $i $EVAL_MODE             2>&1 | tee log/$EVAL_MODE/$MODEL_BASE_NAME/cross_mmlu_p$i.log

done




for ((i=1; i<=1; i++))
do

    bash scripts/eval.sh cross_logiqa $MODEL_NAME $GPU $BZ $i $EVAL_MODE           2>&1 | tee log/$EVAL_MODE/$MODEL_BASE_NAME/cross_logiqa_p$i.log

done
#################################################
#################################################
echo "Finished"