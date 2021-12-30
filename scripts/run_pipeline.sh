#!/bin/bash

CLUSTER_COUNT=30
TOTAL_FILES=30
CLUSTER_PREFIX=Yasaman
BASE=$(pwd)
LOG_DIR=$BASE/Logs
mkdir -p $LOG_DIR
rm -rf $LOG_DIR/*
#$BASE/create_clusters.sh $CLUSTER_COUNT $CLUSTER_PREFIX
$BASE/run_files.sh $TOTAL_FILES
#$BASE/terminate_clusters.sh