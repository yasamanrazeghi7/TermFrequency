#!/bin/bash

CLUSTER_NAME=$1
CLUSTER_INFO_FILE=$2
TMP_FILE=$(mktemp /tmp/XXXX-aws-cluster.XXXXXX)
aws emr create-cluster \
--name "$CLUSTER_NAME" \
--release-label emr-5.33.1 \
--applications Name=Spark \
--ec2-attributes KeyName=XXXX \
--instance-type m5.xlarge \
--instance-count 3 \
--use-default-roles > $TMP_FILE
CLUSTER_ID=$(cat $TMP_FILE | awk '{print $2}')
echo "Cluster is created with id: $CLUSTER_ID"

while true; do
  if aws emr describe-cluster --cluster-id "$CLUSTER_ID" | grep "STATUS" | grep -q "WAITING"; then
    echo "It's ready"
    break
  fi
  echo "Waiting for the cluster to be ready..."
  sleep 30
done
#CLUSTER_ID="j-12E10SE75PL2"
#CLUSTER_ID="j-33FE9ING7YVZA"
aws emr describe-cluster --cluster-id "$CLUSTER_ID" > $TMP_FILE
#echo $TMP_FILE
DOMAIN=$(head -n 1 $TMP_FILE | awk '{print $6}')
echo "Domain is: $DOMAIN"
echo "$CLUSTER_ID $DOMAIN"  >> $CLUSTER_INFO_FILE

