LISTEN_FOLDER=$1


while :
do 
    echo "Checking new models missing evaluation... for $LISTEN_FOLDER every 60 minutes"
    echo time is `date`

    for dir in $LISTEN_FOLDER/*; 
    do 
        BASENAME=`basename "$dir"` 

        if [[ $BASENAME == checkpoint-* && $BASENAME != *results ]];
        then
            
            echo $BASENAME            
            FULLPATH=$LISTEN_FOLDER/$BASENAME-results
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
                qsub -v "MODEL_INDEX=$BASENAME,LISTEN_FOLDER=$LISTEN_FOLDER" -o "log/${MODEL_NAME}_${BASENAME}.log" -N "${MODEL_NAME}_${BASENAME}" scripts/convert_and_evaluate.sh


            fi

        fi

    done

    echo time is `date`
    echo 'Sleeping for 60 minutes'
    sleep 3600

done

