# RayServe on Cloud TPUs
We provide an example that showcases how to serve a Diffusion model on 
TPU VMs using [RayServe](https://docs.ray.io/en/latest/serve/index.html).

Note - this is still a WIP.

## How it Works

`ray_serve_diffusion_flax.py` uses the [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4#jaxflax) model and FastAPI to build the example.

The model server is composed of `APIIngress` which routes requests containing prompts to the TPU-backed model server.

The overall example contains a YAML usable by Ray's cluster launcher and autoscaler + RayServe 

## Model Server Spin Up

First, install the required packages (either on a VM or laptop):
```
$ pip3 install -r requirements-serve.txt
```

Note: If you would like to develop from a VM, you can use
```
./create_cpu.sh
```
as you could in previous Ray on Cloud TPU examples.

### Starting the Ray Cluster

We utilize the [Ray Cluster Launcher](https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-cli.html) for launching the Ray cluster:
```
$ ray up -y ray_serve_diffusion_tpu.yaml
```

### Pulling up the Ray Dashboard
Once the Ray cluster is up, you can connect to the Ray dashboard with the following command:
```
$ ray dashboard ray_serve_diffusion_tpu.yaml 
...
2023-07-10 16:19:24,064	INFO log_timer.py:25 -- NodeUpdater: ray-ray-serve-diffusion-head-523354b9-compute: Got IP  [LogTimer=0ms]
2023-07-10 16:19:24,064	INFO command_runner.py:343 -- Forwarding ports
2023-07-10 16:19:24,064	VINFO command_runner.py:347 -- Forwarding port 8265 to port 8265 on localhost.
2023-07-10 16:19:24,064	VINFO command_runner.py:371 -- Running `None`
2023-07-10 16:19:24,064	VVINFO command_runner.py:373 -- Full command is `ssh -tt -L 8265:localhost:8265 -i <REDACTED> pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=/tmp/ray_ssh_559623ff5c/a39f283bdb/%C -o ControlPersist=10s -o ConnectTimeout=120s ubuntu@35.186.59.139 while true; do sleep 86400; done`
```
As shown above, this port forwards port 8265 from the Ray head node. You can then open the Ray dashboard locally at http://localhost:8265.

### Monitoring the Ray Cluster/Autoscaler
`ray_serve_diffusion_tpu.yaml` specifies `min_workers: 1`, e.g. that at least one `ray_tpu` worker should be up at a given time.

The autoscaler makes calls against the GCE backend (similar to running `gcloud ...`) and may fail in case of malformed requests or out of quota errors. In order to see the status of the autoscaler, you can run the following command to stream the logs:

```
$ ray monitor ray_serve_diffusion_tpu.yaml
```

To get information about the Ray cluster, you can also connect to an interactive environment on the Ray head node with
```
$ ray attach ray_serve_diffusion_tpu.yaml
```

From there, you can poll the status of the Ray cluster:
```
ubuntu@ray-ray-serve-diffusion-head-523354b9-compute:~$ ray status
======== Autoscaler status: 2023-07-10 17:02:05.760135 ========
Node status
---------------------------------------------------------------
Healthy:
 1 ray_head_default
 1 ray_tpu
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/244.0 CPU
 0.0/1.0 TPU
 0B/287.66GiB memory
 0B/123.88GiB object_store_memory

Demands:
 (no resource demands)
```

### Deploying the Model Server
Once your Ray cluster launcher is up and running, you can start up the model servers using the following command:

```
$ serve run --working-dir="./" --address=ray://${RAY_HEAD_IP}:10001 -h 0.0.0.0 -p 8000 ray_serve_diffusion_flax:deployment 
```

Note that you will need to replace `${RAY_HEAD_IP}` with the Ray head node's *internal* IP address.

Ray currently does not support functionality for getting the internal IP address, but we can extract this as follows:
```
$ EXTERNAL_IP=$(ray get-head-ip ray_serve_diffusion_tpu.yaml)
$ gcloud compute instances list | grep $EXTERNAL_IP
ray-ray-serve-diffusion-head-523354b9-compute        us-central2-b  n1-standard-4                 10.130.0.9    35.186.59.139    RUNNING
```
In this example, `RAY_HEAD_IP` should be set using 10.130.0.9.

Note: Make sure $RAY_HEAD_ADDRESS is not set.

### Stopping the Model Server
During development, you might want to shutdown and restart the model server, e.g. in cases where your model server hits a hard error that it cannot recover from quickly.

To easily do that, you can attach to your head node (`ray attach`) and run the following command to explicitly shutdown the Ray Serve session:

```
$ serve shutdown -y
```

Afterwards, you can restart the Ray Serve model servers as before.


### Tearing down the cluster

```
$ ray down -y ray_serve_diffusion_tpu.yaml
```


## Running on GKE
TODO

## Load Testing
For convenience, we provide a script, `fake_load_test.py`, that can be used to send prompts to the model server.

Usage:
```
$ python3 fake_load_test.py --ip=${RAY_HEAD_IP}
```

Flags:
- `ip`: the internal IP of the Ray HEAD address. You should be able to access the external IP as well, but you will need to make sure that your GCE firewall settings allows this.
- `num_requests`: The number of requests to send in total
- `save_pictures`: Whether or not to save as an image. If set to true, this is saved at `diffusion_results.png`.
- `batch_size`: The number of requests to send at a time.

## Caveats
- As of right now, autoscaling does not work out of the box
- This example has only been tested on single host TPU VMs (e.g. v4-8)
- This example, as of now, is a proof of concept demo and has not been comprehensively benchmarked for optimal performance.
