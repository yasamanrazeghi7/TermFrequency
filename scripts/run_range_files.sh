#!/bin/bash

DOMAIN_NAME=$1
BEGIN_INDEX=$2
END_INDEX=$3
ALL_NUMBERS=(`seq -w 0 29`)
ADDR=hadoop@$DOMAIN_NAME
PEM_FILE=~/yrazeghi.pem
ssh  -i ~/yrazeghi.pem $ADDR "python3 -m pip install boto3"
scp -i ~/yrazeghi.pem ../dataset_preproccessor.py "$ADDR:~"
scp -i ~/yrazeghi.pem ../num_counter.py "$ADDR:~"
for i in `seq -w $BEGIN_INDEX $END_INDEX`; do
  file_index=${ALL_NUMBERS[$i]}
  echo "Running file $file_index"
  ssh  -i ~/yrazeghi.pem $ADDR "python3 ~/dataset_preproccessor.py pile/$file_index.jsonl"
done

echo "My job is done [$BEGIN_INDEX..$END_INDEX]"