#!/bin/bash

BASE=$(pwd)

## Need to assign
CLUSTER_COUNT=30
TOTAL_FILES=30
CLUSTER_PREFIX=NumCounter #name each cluster on aws
#RUN_RANGE_FILE=$BASE/run_dummy_range_files.sh

## This is a code to run on each file
RUN_RANGE_FILE=$BASE/run_num_occurrence_range_files.sh
## This is a code that aggregates all the files results
CLEANUP_FILE=$BASE/cleanup_num_occurrence.sh

CLUSTER_INFO_FILE=$BASE/ClusterInfo.txt

LOG_DIR=$BASE/Logs
mkdir -p $LOG_DIR
rm -rf $LOG_DIR/*
$BASE/create_clusters.sh $CLUSTER_COUNT $CLUSTER_PREFIX
$BASE/run_files.sh $TOTAL_FILES $RUN_RANGE_FILE
# Cleanup
DOMAINS=$(cat $CLUSTER_INFO_FILE | awk '{print $2}')
DOMAINS=($DOMAINS)
DOMAIN=${DOMAINS[0]}
echo "Running cleanup on Domain: $DOMAIN"
bash -c "$CLEANUP_FILE $DOMAIN" &> $LOG_DIR/cleanup.log

##terminates all the clusters
$BASE/terminate_clusters.sh