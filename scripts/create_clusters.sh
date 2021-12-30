#!/bin/bash

CLUSTER_COUNT=$1
CLUSTER_PREFIX=${2:-YASAMAN}
BASE=$(pwd)
LOG_DIR=$BASE/Logs
mkdir -p $LOG_DIR
CLUSTER_INFO_FILE=$BASE/ClusterInfo.txt
echo -n "" > $CLUSTER_INFO_FILE
for (( c=1; c<=$CLUSTER_COUNT; c++ ))
do
  echo "Create cluster $c..."
	$BASE/create_single_cluster.sh "$CLUSTER_PREFIX"_$c $CLUSTER_INFO_FILE > $LOG_DIR/create_logs_$c.txt &
	sleep 5
done

FAIL=0
for job in `jobs -p`
do
  echo "Waiting for job $job to be finished"
  wait $job || let "FAIL+=1"
done

if [ "$FAIL" == "0" ];
then
  echo "All jobs are done!"
else
  echo "There are ($FAIL) failed jobs!"
fi