#!/bin/bash

CLUSTER_COUNT=30
TOTAL_FILES=30
CLUSTER_PREFIX=WordNumCounter
BASE=$(pwd)
CLUSTER_INFO_FILE=$BASE/ClusterInfo.txt
#RUN_RANGE_FILE=$BASE/run_dummy_range_files.sh
RUN_RANGE_FILE=$BASE/run_word_num_occurrence_range_files.sh
CLEANUP_FILE=$BASE/cleanup_word_num_occurrence.sh
LOG_DIR=$BASE/Logs
mkdir -p $LOG_DIR
rm -rf $LOG_DIR/*
#$BASE/create_clusters.sh $CLUSTER_COUNT $CLUSTER_PREFIX
$BASE/run_files.sh $TOTAL_FILES $RUN_RANGE_FILE
# Cleanup
DOMAINS=$(cat $CLUSTER_INFO_FILE | awk '{print $2}')
DOMAINS=($DOMAINS)
DOMAIN=${DOMAINS[0]}
bash -c "$CLEANUP_FILE $DOMAIN" &> $LOG_DIR/cleanup.log
$BASE/terminate_clusters.sh