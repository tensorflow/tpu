# Ray on Cloud TPU examples

This folder contains minimal examples of how to use Ray (ray.io)
with Cloud TPUs.

Our objective is to bring the native experience of Ray to Cloud TPU
users. These examples serve as a reference point for you to get started.


## Folder Structure
 - [serve](src/serve/) - examples using RayServe.
 - [tune](src/tune/) - examples using RayTune.
 - `create_tpu_service_account.sh` - convenience script to create a
   service account with TPU admin access. 
 - `create_cpu.sh` - convenience script to spin up a dev node on GCP.
 - `deploy_to_admin.sh` - convenience script to `rsync` code to your dev node.

## Getting Started
To create a service account that has TPU VM admin access:
```
./create_tpu_serivce_account.sh
```
This will create a service account named `tpuAdmin` with the following
roles:
- `roles/tpu.admin`
- `roles/iam.serviceAccountUser`

To create a dev node on GCP (e.g. `n1-standard-1`):
```
./create_cpu.sh
```
This will create a CPU VM of name `$USER-dev`.

To sync code to your dev node:
```
./deploy_to_admin.sh
```
This will SCP code within [src](src/) to the dev machine.

Once your VM is deployed with starter code, SSH to the machine and install
the requirements:
```
$ gcloud compute ssh $USER-dev -- -L8265:localhost:8265
$ pip install -r src/requirements.txt
```

## Support

- [x] Single host TPU VM examples
  - [x] RayServe examples (see [serve/](src/serve/))
  - [x] RayTune examples (see [tune/](src/tune/))
- [ ] Multi host TPU VM examples
  - [ ] RayServe examples
  - [ ] RayTune examples
  - [ ] RayTrain examples
