#!/bin/bash

DOMAIN_NAME=$1
BEGIN_INDEX=$2
END_INDEX=$3
ALL_NUMBERS=(`seq -w 0 29`)
ADDR=hadoop@$DOMAIN_NAME
PEM_FILE=~/yrazeghi.pem
for i in `seq -w $BEGIN_INDEX $END_INDEX`; do
  file_index=${ALL_NUMBERS[$i]}
  echo "Running dummy file $file_index"
done
echo "My job is done [$BEGIN_INDEX..$END_INDEX]"