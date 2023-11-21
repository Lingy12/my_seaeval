EXP_OUT_DIR=$1 # listen to experiment folder


    echo time is `date`

    for dir in $EXP_OUT_DIR/*; 
    do 
        BASENAME=`basename "$dir"` 

        if [[ $BASENAME != "model_archive" ]];
        then
            echo $BASENAME
            # echo $BASENAME            
            # FULLPATH=$LISTEN_FOLDER/$BASENAME-results
            # echo 'Checking' $FULLPATH
            count=`ls -1 $BASENAME/*results 2>/dev/null | wc -l`
            # echo $count
            if [ count == 0 ]; 
            then
                echo "$EXP_OUT_DIR"
                echo "$EXP_OUT_DIR/$BASENAME"

            fi

        fi

    done

