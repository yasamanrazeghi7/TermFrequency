#!/bin/bash
BASE=$(pwd)
LOG_DIR=$BASE/Logs
mkdir -p $LOG_DIR
TOTAL_FILES=$1
RUN_RANGE_FILE=$2
CLUSTER_INFO_FILE=$BASE/ClusterInfo.txt
CLUSTER_COUNT=$(cat $CLUSTER_INFO_FILE | wc -l)
if [ $CLUSTER_COUNT -eq "0" ]; then
  echo "No cluster found!"
  exit;
fi
CLUSTER_SHARE=$((TOTAL_FILES / CLUSTER_COUNT))
if [ $((TOTAL_FILES % CLUSTER_COUNT)) -ne "0" ]; then
  CLUSTER_SHARE=$((CLUSTER_SHARE + 1))
fi
DOMAINS=$(cat $CLUSTER_INFO_FILE | awk '{print $2}')
DOMAINS=($DOMAINS)
BEGIN_INDEX=0
for (( c=1; c<$CLUSTER_COUNT; c++ ))
do
  DOMAIN=${DOMAINS[$((c - 1))]}
  echo "Sending jobs [$BEGIN_INDEX..$((BEGIN_INDEX + CLUSTER_SHARE - 1))] to cluster $c with domain $DOMAIN..."
  bash -c "$RUN_RANGE_FILE $DOMAIN $BEGIN_INDEX $((BEGIN_INDEX + CLUSTER_SHARE - 1)) &> $LOG_DIR/run_log_$c.txt" &
  BEGIN_INDEX=$((BEGIN_INDEX + CLUSTER_SHARE))
  sleep 1
done

# The last cluster covers all remaining files
DOMAIN=${DOMAINS[$((CLUSTER_COUNT - 1))]}
echo "Sending jobs [$BEGIN_INDEX..$((TOTAL_FILES - 1))] to cluster $CLUSTER_COUNT with domain $DOMAIN..."
bash -c "$RUN_RANGE_FILE $DOMAIN $BEGIN_INDEX $((TOTAL_FILES - 1)) &> $LOG_DIR/run_log_$CLUSTER_COUNT.txt" &

FAIL=0
for job in `jobs -p`
do
  echo "Waiting for run-file job $job to be finished"
  wait $job || let "FAIL+=1"
done

if [ "$FAIL" == "0" ];
then
  echo "All run-files jobs are done!"
else
  echo "There are ($FAIL) failed jobs!"
fi