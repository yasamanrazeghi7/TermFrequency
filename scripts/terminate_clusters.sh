#!/bin/bash

BASE=$(pwd)
CLUSTER_INFO_FILE=$BASE/ClusterInfo.txt
CLUSTER_IDS=$(cat $CLUSTER_INFO_FILE | awk '{print $1}')
for CLUSTER_ID in $CLUSTER_IDS
do
  echo "Terminating cluster $CLUSTER_ID"
  aws emr terminate-clusters --cluster-id "$CLUSTER_ID"
  sleep 3
done
echo "Done!"