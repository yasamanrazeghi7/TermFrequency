#!/bin/bash

DOMAIN_NAME=$1
BEGIN_INDEX=$2
END_INDEX=$3
ALL_NUMBERS=(`seq -w 0 29`)
ADDR=hadoop@$DOMAIN_NAME
PEM_FILE=~/XXXX.pem
WORKSPACE_PATH=`realpath ..`
scp -i $PEM_FILE -r $WORKSPACE_PATH/SparkScripts "$ADDR:~"
scp -i $PEM_FILE $WORKSPACE_PATH/*.py "$ADDR:~"
for i in `seq -w $BEGIN_INDEX $END_INDEX`; do
  file_index=${ALL_NUMBERS[$i]}
  echo "Running counting word num in file $file_index"
  ## assign input outputs of the code
  input_file_pattern="s3://pilebucketXXXX/pile/new_processed/${file_index}_*.jsonl"
  output_folder="s3://pilebucketXXXX/results/word_1num_counting/${file_index}/"
  window_size=5
  num_count=1
  ## run the code remotely with inputs
  ssh  -i $PEM_FILE $ADDR "spark-submit ~/SparkScripts/word_num_occurrence.py '$input_file_pattern' '$output_folder' $window_size $num_count"
done
echo "My job is done [$BEGIN_INDEX..$END_INDEX]"