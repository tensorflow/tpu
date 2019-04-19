# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/bin/bash
set -x
RUNDIR="$(cd "$(dirname "$0")" ; pwd -P)"

# Setup configuration
export NUM_TPU=3
export ACCELERATOR_TYPE='v2-8'
export MODEL_BASE_PATH=""
export MODEL_NAME=""
export PROJECT_NAME=""
export POOL_NAME=""
export ZONE="us-central1-b"
export REGION="us-central1"

# Used when cleaning up resources
export RESOURCE_LIST="$POOL_NAME-resources.txt"

function err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $@" >&2
  exit 1
}

function validate_arg_name() {
  if [ -z "$1" ]; then
    err "Variable $2 cannot be empty."
  fi
}

function validate_args() {
  validate_arg_name "${ACCELERATOR_TYPE}" "ACCELERATOR_TYPE"
  validate_arg_name "${MODEL_BASE_PATH}" "MODEL_BASE_PATH"
  validate_arg_name "${MODEL_NAME}" "MODEL_NAME"
  validate_arg_name "${PROJECT_NAME}" "PROJECT_NAME"
  validate_arg_name "${POOL_NAME}" "POOL_NAME"
  validate_arg_name "${ZONE}" "ZONE"
  validate_arg_name "${REGION}" "REGION"
}

function gen_cidr_block() {
  printf "10.%s.%s.0" $(( $RANDOM % 255 )) $(( $RANDOM % 255 ))
}

function gen_base_name() {
  printf "%s-tpu-infer-%s-%s" ${POOL_NAME?} $(date +"%m%d%H%M%S" --utc) $(( $RANDOM % 10000 ))
}

function check_tpu_tf_versions() {
  gcloud compute tpus versions list | grep ^nightly$ > /dev/null
  if [[ $? -ne 0 ]]; then
    err "You must be whitelisted to nightly TPU version for TPU Inference."
  fi
}

function check_model_exists() {
  num_versions=$(expr $(gsutil ls ${MODEL_BASE_PATH?} | wc | awk '{print $1}')-1)
  if [[ $num_versions < 1 ]]; then
    err "The MODEL_BASE_PATH provided is not valid."
  fi
}

function setup_vm_tpu_pair() {
  echo "Creating Pair ($1, $2)"
  vm_name=$1
  tpu_name=$2

  # Create TPU
  for i in {1..3}  # retries for cases like cidr conflict
  do
    cidr=$(gen_cidr_block)
    echo "cidr==$cidr"
    gcloud alpha compute tpus create "$tpu_name" \
      --network=default \
      --accelerator-type=${ACCELERATOR_TYPE?} \
      --range=$cidr \
      --version=nightly \
      --model-base-path=${MODEL_BASE_PATH?} \
      --model-name=${MODEL_NAME?}
    if [[ $? -eq 0 ]]; then
      echo "tpus = $tpu_name" >> $RESOURCE_LIST
      break
    fi
  done
  tpu_ip_address=$(gcloud compute tpus describe "$tpu_name" \
    --zone=$ZONE | grep '^ipAddress: ' | awk '{print $2}')
  echo "tpu_ip_address==$tpu_ip_address"

  # Create Client VM
  service_name="$POOL_NAME.endpoints.$PROJECT_NAME.cloud.goog"
  gcloud compute instances create "$vm_name" \
    --zone=$ZONE \
    --machine-type=n1-standard-2 \
    --maintenance-policy=MIGRATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=http-server,https-server \
    --image-project=ml-images \
    --image-family=tf-nightly \
    --boot-disk-size=10GB \
    --boot-disk-type=pd-standard \
    --boot-disk-device-name="$vm_name" \
    --metadata startup-script="\
    # Run ESP container on VM
    #   - vm:80 <> espdocker:9000 (/healthz) <> tpu:8473 (REST)
    sudo docker network create --driver bridge esp_net;
    docker run --detach --name=esp --volume=/home:/esp \
      --publish=80:9000 \
      --net=esp_net \
      gcr.io/endpoints-release/endpoints-runtime:1 \
      --service=$service_name \
      --rollout_strategy=managed \
      --http_port=9000 \
      --backend=http://$tpu_ip_address:8473 \
      --worker_processes=auto \
      -z healthz;" && echo "instances = $vm_name" >> $RESOURCE_LIST
}

function setup_inference_pool() {
  echo "┌─────────────────────────────────┐"
  echo "│   Google Cloud TPU Inference    │"
  echo "└─────────────────────────────────┘"
  echo " Setup starting..."
  check_tpu_tf_versions
  check_model_exists
  echo "# Resources created for = $POOL_NAME" >> $RESOURCE_LIST

  setup_endpoints

  # Create VM-TPU Pool
  for (( i=0; i<${NUM_TPU?}; i++ ))
  do
    base_name=$(gen_base_name)
    vms+=("$base_name-vm")
    setup_vm_tpu_pair "${base_name}-vm" "${base_name}-tpu" &
  done

  # Wait for endpoints and pool creation
  for job in `jobs -p`
  do
    wait $job
  done

  vms=$( IFS=$','; echo "${vms[*]}" )
  setup_l7lb $vms
  enable_firewall
}

function cleanup() {
  echo "Cleaning up pool..."

  tpus=()
  endpoints=()
  instances=()
  instance_groups=()
  backend_services=()
  health_checks=()
  url_maps=()
  target_http_proxies=()
  addresses=()
  forwarding_rules=()
  while read -r line; do
    resource_type=$( echo $line | awk '{print $1}' )
    resource_name=$( echo $line | awk '{print $3}' )
    case "$resource_type" in
      "tpus") tpus+=($resource_name) ;;
      "endpoints") endpoints+=($resource_name) ;;
      "instances") instances+=($resource_name) ;;
      "instance-groups") instance_groups+=($resource_name) ;;
      "backend-services") backend_services+=($resource_name) ;;
      "health-checks") health_checks+=($resource_name) ;;
      "url-maps") url_maps+=($resource_name) ;;
      "target-http-proxies") target_http_proxies+=($resource_name) ;;
      "addresses") addresses+=($resource_name) ;;
      "forwarding-rules") forwarding_rules+=($resource_name) ;;
      *) echo "Unkonwn resource type: $resource_type" ;;
    esac
  done < $RESOURCE_LIST

  for tpu in "${tpus[@]}"
  do
    gcloud compute tpus delete $tpu --zone $ZONE --async
  done

  for endpoint in "${endpoints[@]}"
  do
    gcloud endpoints services delete $endpoint
  done

  gcloud compute instances delete ${instances[*]}&

  for address in "${addresses[@]}"
  do
    gcloud compute addresses delete $address --global
  done

  for forwarding_rule in "${forwarding_rules[@]}"
  do
    gcloud compute forwarding-rules delete $forwarding_rule --global
  done

  for target_http_proxy in "${target_http_proxies[@]}"
  do
    gcloud compute target-http-proxies delete $target_http_proxy
  done

  for url_map in "${url_maps[@]}"
  do
    gcloud compute url-maps delete $url_map
  done

  for backend_service in "${backend_services[@]}"
  do
    gcloud compute backend-services delete $backend_service --global
  done

  for health_check in "${health_checks}"
  do
    gcloud compute health-checks delete $health_check
  done

  for instance_group in "${instance_groups[@]}"
  do
    gcloud compute instance-groups unmanaged delete $instance_group
  done
}

function setup_endpoints() {
  sed -e "s/<MY_PROJECT_ID>/$PROJECT_NAME/" \
    -e "s/<ENDPOINT_NAME>/$POOL_NAME/" \
    -e "s/<MODEL_NAME>/$MODEL_NAME/" openapi.yaml > /tmp/openapi.yaml
  service_name="$POOL_NAME.endpoints.$PROJECT_NAME.cloud.goog"
  gcloud endpoints services deploy /tmp/openapi.yaml && \
      echo "endpoints = $service_name" >> $RESOURCE_LIST
}

function setup_l7lb() {
  vms=$1

  HEALTH_CHECK="$POOL_NAME-health-check"
  BACKEND="$POOL_NAME-backend"
  INSTANCE_GROUP="$POOL_NAME-instance-group"
  URL_MAP="$POOL_NAME-map"
  L7LB="$POOL_NAME-l7lb"
  IP_NAME="$POOL_NAME-ip"
  FORWARDING_RULE="$POOL_NAME-forwarding-rule"

  # Create health check
  gcloud compute health-checks create http $HEALTH_CHECK \
      --request-path /healthz \
      --port 80 && \
      echo "health-checks = $HEALTH_CHECK" >> $RESOURCE_LIST

  # Create backend
  gcloud compute backend-services create $BACKEND \
      --protocol=HTTP \
      --health-checks $HEALTH_CHECK \
      --global && \
      echo "backend-services = $BACKEND" >> $RESOURCE_LIST

  # Create and populate unmanaged instance group
  gcloud compute instance-groups unmanaged create $INSTANCE_GROUP && \
      echo "instance-groups = $INSTANCE_GROUP" >> $RESOURCE_LIST
  gcloud compute instance-groups unmanaged add-instances $INSTANCE_GROUP \
      --instances=$vms

  # Register unmanaged instance group to backend
  gcloud compute backend-services add-backend $BACKEND \
      --balancing-mode UTILIZATION \
      --instance-group $INSTANCE_GROUP \
      --instance-group-zone $ZONE \
      --global

  # Create url map for backend service
  gcloud compute url-maps create $URL_MAP --default-service $BACKEND && \
      echo "url-maps = $URL_MAP" >> $RESOURCE_LIST

  # Create l7lb
  gcloud compute target-http-proxies create $L7LB --url-map $URL_MAP && \
      echo "target-http-proxies = $L7LB" >> $RESOURCE_LIST

  # Create public IP for l7lb
  gcloud compute addresses create $IP_NAME --ip-version=IPV4 --global && \
      echo "addresses = $IP_NAME" >> $RESOURCE_LIST

  # Forwarding rule for public IP -> L7LB request forwarding
  IP=$(gcloud compute addresses list | grep "$POOL_NAME-ip" | awk '{print $2}')
  gcloud compute forwarding-rules create $FORWARDING_RULE \
    --address $IP \
    --global \
    --target-http-proxy "$POOL_NAME-l7lb" \
    --ports 80 && \
    echo "forwarding-rules = $FORWARDING_RULE" >> $RESOURCE_LIST
}

function help() {
    echo "(Help) Usage: $0 {cleanup|setup|enable_firewall|firewall_status}"
}

function enable_firewall() {
  echo "Enabling firewall..."
  gcloud compute firewall-rules create www-firewall-80 \
      --target-tags http-server --allow tcp:80
}

function firewall_status() {
  gcloud compute firewall-rules list
}

function main() {
  pushd $RUNDIR > /dev/null

  validate_args || exit 1
  gcloud config set project $PROJECT_NAME
  gcloud config set compute/region $REGION
  gcloud config set compute/zone $ZONE

  case "$1" in
    cleanup)
    echo "Are you sure you want to delete your TPU inference pool?"
    echo "Deleting: POOL_NAME == $POOL_NAME ?"
      select yn in "Yes" "No"; do
        case $yn in
            Yes ) cleanup; break;;
            No ) exit;;
        esac
      done;;
    setup) setup_inference_pool;;
    enable_firewall) enable_firewall;;
    firewall_status) firewall_status;;
    *) help ;;
  esac

  popd > /dev/null
}

main $1
