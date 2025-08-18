#! /bin/bash

# Creates a new GPU instance, transfers the code to it, and runs some installation steps.

export ZONE="europe-southwest1-a"
export INSTANCE_NAME="cs285"

echo "Accessing instance..."

while true; do
  output=$(gcloud compute ssh $INSTANCE_NAME --zone=$ZONE 2>&1)

  if [ $? -eq 0 ]; then
    echo $output
    break
  else
    sleep 10
  fi
done

echo "-------------------------------------"
echo "Transferring files to instance..."


echo "-------------------------------------"
echo "Running setup..."

