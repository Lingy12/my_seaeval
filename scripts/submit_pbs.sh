


#bash automatic_checking.sh 2>&1 | tee automatic_checking.log




#qsub pbs_jobs/224.sh
#qsub pbs_jobs/108.sh
#qsub pbs_jobs/67.sh
#qsub pbs_jobs/223.sh


#<<COMMENT


#for ((i=229; i<=234; i++))
#do
#    qsub pbs_jobs/$i.sh
#done

#COMMENT


<<COMMENT

for ((i=4415652; i<=4415657; i++))
do
    qdel $i.pbs101
done

COMMENT



#qsub -v "MODEL_INDEX=checkpoint-50" convert_only.sh  


for ((i=200; i<=6350; i=i+200))
do

    BASENAME=checkpoint-$i
    qsub -v "MODEL_INDEX=$BASENAME" convert_and_evaluate.sh

done


