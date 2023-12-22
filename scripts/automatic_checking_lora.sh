LISTEN_FOLDER=$1


echo "Checking new models missing evaluation... for $LISTEN_FOLDER"
echo time is `date`

for dir in $LISTEN_FOLDER/*; 
do 
    BASENAME=`basename "$dir"` 

    if [[ $BASENAME == checkpoint-* && $BASENAME != *results ]];
    then
        
        echo $BASENAME            
        FULLPATH=$LISTEN_FOLDER/converted_checkpoint/$BASENAME-results
        echo 'Checking' $FULLPATH

        if [ ! -d "$FULLPATH" ]; 
        then

            echo "=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  ="
            echo "$FULLPATH does not exist."
            mkdir -p $FULLPATH
            MODEL_NAME=$(basename $LISTEN_FOLDER)
            # Do Conversion and Evaluation
            echo "Converting and evaluating $LISTEN_FOLDER/$BASENAME"
            mkdir -p log
            qsub -v "MODEL_INDEX=$BASENAME,LISTEN_FOLDER=$LISTEN_FOLDER" -o "log/${MODEL_NAME}_${BASENAME}.log" -N "${MODEL_NAME}_${BASENAME}" scripts/convert_lora_and_evaluate.sh


        fi

    fi

done
echo "$LISTEN_FOLDER finished"
