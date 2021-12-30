#!/bin/bash

DOMAIN_NAME=$1
ADDR=hadoop@$DOMAIN_NAME
PEM_FILE=~/yrazeghi.pem
WORKSPACE_PATH=`realpath ..`
scp -i $PEM_FILE -r $WORKSPACE_PATH/SparkScripts "$ADDR:~"
scp -i $PEM_FILE $WORKSPACE_PATH/*.py "$ADDR:~"
echo "Cleanup counting word num"
input_folder_pattern="s3://pilebucketyasaman/results/word_1num_counting/*/part-*"
local_output_path="~/aggregated_results"
s3_output_path="s3://pilebucketyasaman/results/word_1num_counting/aggregated/"
ssh -i $PEM_FILE $ADDR "mkdir -p $local_output_path"
ssh -i $PEM_FILE $ADDR "spark-submit ~/SparkScripts/aggregate_results.py $input_folder_pattern $local_output_path"
ssh -i $PEM_FILE $ADDR "aws s3 cp --recursive $local_output_path $s3_output_path"
echo "My cleanup is done!"