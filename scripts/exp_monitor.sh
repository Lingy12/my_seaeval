EXP_FOLDER=$1

echo "Checking new models missing evaluation... for $EXP_FOLDER"
echo time is `date`

while true
do
    for dir in $EXP_FOLDER/*; 
    do 
        BASENAME=`basename "$dir"` 
        echo "============================================================================"
        echo "Running scanning for $BASENAME"

        bash scripts/automatic_checking.sh $dir & # running in parallel

    done

    sleep 30
    echo "Sleep 30s"
done 
