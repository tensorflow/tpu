# RayServe on Cloud TPUs
We provide an example that showcases how to serve a Diffusion model on 
TPU VMs using [RayServe](https://docs.ray.io/en/latest/serve/index.html).

## How it Works

`ray_serve_diffusion_flax.py` uses the [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4#jaxflax)
model and FastAPI to build the example.

The model server is composed of `APIIngress` which routes requests containing
prompts to the TPU-backed model server.

## Starting your Ray cluster
Before starting, make sure you change your `project_id` within `cluster.yaml`
to your GCP project and that your GCP project has (1) the TPU API enabled, and
(2) proper TPU quotas granted.

Navigate to this folder:

```
$ cd src/serve
```

and make sure you have the requirements installed:

```
$ pip3 install -r requirements.txt
```

Then start your Ray cluster as follows:

```
$ ray up -y cluster.yaml
Cluster: ray-serve-diffusion

Checking GCP environment settings
...

2023-08-25 15:54:24,083	INFO node.py:311 -- wait_for_compute_zone_operation: Waiting for operation operation-1692978863799-603c15bc9fcd0-16e91745-2eb95a63 to finish...
2023-08-25 15:54:29,257	INFO node.py:330 -- wait_for_compute_zone_operation: Operation operation-1692978863799-603c15bc9fcd0-16e91745-2eb95a63 finished.
  New status: up-to-date

Useful commands
  Monitor autoscaling with
    ray exec /home/$USER/src/serve/cluster.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
  Connect to a terminal on the cluster head:
    ray attach /home/$USER/src/serve/cluster.yaml
  Get a remote shell to the cluster manually:

```

### Pulling up the Ray Dashboard
Once the Ray cluster is up, you can connect to the Ray dashboard with the following command:

```
$ ray dashboard cluster.yaml 
...
2023-07-10 16:19:24,064	INFO log_timer.py:25 -- NodeUpdater: ray-ray-serve-diffusion-head-523354b9-compute: Got IP  [LogTimer=0ms]
2023-07-10 16:19:24,064	INFO command_runner.py:343 -- Forwarding ports
2023-07-10 16:19:24,064	VINFO command_runner.py:347 -- Forwarding port 8265 to port 8265 on localhost.
2023-07-10 16:19:24,064	VINFO command_runner.py:371 -- Running `None`
2023-07-10 16:19:24,064	VVINFO command_runner.py:373 -- Full command is `ssh -tt -L 8265:localhost:8265 -i <REDACTED> pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=/tmp/ray_ssh_559623ff5c/a39f283bdb/%C -o ControlPersist=10s -o ConnectTimeout=120s ubuntu@35.186.59.139 while true; do sleep 86400; done`
```
As shown above, this port forwards port 8265 from the Ray head node. You can then open the Ray dashboard locally at http://localhost:8265.

### Monitoring the Ray Cluster/Autoscaler
`cluster.yaml` specifies `min_workers: 1`, e.g. that at least one `ray_tpu` worker
should be up at a given time.

The autoscaler makes calls against the GCE backend (similar to running
`gcloud ...`) and may fail in case of malformed requests or out of quota errors.
In order to see the status of the autoscaler, you can run the following command
to stream the logs:

```
$ ray monitor cluster.yaml
...
======== Autoscaler status: 2023-08-25 15:48:02.454358 ========
Node status
---------------------------------------------------------------
Healthy:
 1 ray_head_default
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/4.0 CPU
 0B/8.30GiB memory
 0B/4.15GiB object_store_memory

Demands:
 (no resource demands)
2023-08-25 15:48:02,455	INFO autoscaler.py:594 -- StandardAutoscaler: Terminating the node with id projects/googles-secret-dev-project/locations/us-central2-b/nodes/ray-ray-serve-diffusion-worker-b9a8d2bc-tpu and ip 10.130.0.91. (outdated)
2023-08-25 15:48:02,456	INFO node_provider.py:186 -- NodeProvider: projects/googles-secret-dev-project/locations/us-central2-b/nodes/ray-ray-serve-diffusion-worker-b9a8d2bc-tpu: Terminating node
2023-08-25 15:48:02,537	INFO node.py:563 -- wait_for_tpu_operation: Waiting for operation projects/googles-secret-dev-project/locations/us-central2-b/operations/operation-1692978482497-603c1450fc975-dbf1278b-dcfcecc5 to finish...

...

Resources
---------------------------------------------------------------
Usage:
 0.0/244.0 CPU
 0.0/1.0 TPU
 0B/287.67GiB memory
 0B/123.88GiB object_store_memory

Demands:
 (no resource demands)
2023-08-25 15:54:18,294	INFO autoscaler.py:470 -- The autoscaler took 0.144 seconds to complete the update iteration.

```

To get information about the Ray cluster, you can also connect to an interactive environment on the Ray head node with

```
$ ray attach cluster.yaml
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

### Setting Ray Environment Variables
There are many ways to [interact with a remote Ray Cluster](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html#using-a-remote-cluster).

For convenience, we provide a script that will set `RAY_ADDRESS`
for you:

```
$ source ./set_ray_address.sh 
Make sure that you are running this as source ./set_ray_address.sh
Set RAY_HEAD_IP=10.130.0.157
Set RAY_ADDRESS=http://10.130.0.157:8265
```

### Deploying the Model Server
Once your Ray cluster is up and running and `RAY_ADDRESS` is set, you can start up the model servers using the following command:

```
$ serve run --working-dir="./" --address=ray://${RAY_HEAD_IP}:10001 -h 0.0.0.0 -p 8000 ray_serve_diffusion_flax:deployment 
```

### Stopping the Model Server
If you want to shutdown and restart the model server, you can easily do that
as well.

To easily do that, you can attach to your head node (`ray attach`) and run the following command to explicitly shutdown the Ray Serve session:

```
$ serve shutdown --address=http://${RAY_HEAD_IP}:52365 -y
2023-08-25 18:47:48,243	SUCC scripts.py:609 -- Sent shutdown request; applications will be deleted asynchronously.
```

Confirm that there are no serve instances running:
```
$ serve status --address=http://${RAY_HEAD_IP}:52365
There are no applications running on this cluster.
```

Afterwards, you can restart the Ray Serve model servers as before.


### Tearing down the cluster

```
$ ray down -y cluster.yaml
```

## Load Testing
For convenience, we provide a script, `fake_load_test.py`, that can be used to send prompts to the model server.

Usage:

```
$ python3 fake_load_test.py --ip=${RAY_HEAD_IP}
...
num_requests:  8
batch_size:  8
url:  http://10.130.0.157:8000/imagine
save_pictures:  False
 12%|███████████████████████████████████████████▏     
```

Flags:
- `ip`: the internal IP of the Ray HEAD address. You should be able to access the external IP as well, but you will need to make sure that your GCE firewall settings allows this.
- `num_requests`: The number of requests to send in total
- `save_pictures`: Whether or not to save as an image. If set to true, this is saved at `diffusion_results.png`.
- `batch_size`: The number of requests to send at a time.

By default, `num_requests` and `batch_size` are both set to 8, so 8 requests in
total will be sent at a time.

To test out autoscaling, we suggest increasing both `num_requests` and `batch_size`
to a multiple of 64, which is the batch size we use to target a single model server.

For instance - setting `--batch_size=128 --num_requests=1024` should send
8 batches of 128 and should trigger an event where Ray requests another TPU.

## Autoscaling
Ray serve can trigger autoscaling based on the amount of traffic sent to
a particular load.

From `ray_serve_diffusion_flax.py` we have defined this within the
`autoscaling_config`, i.e.:

```
autoscaling_config={
    "min_replicas": 1,
    "max_replicas": 4,
    "target_num_ongoing_requests_per_replica": _MAX_BATCH_SIZE,
}
```
where `_MAX_BATCH_SIZE` is hard coded to 64. If everything is setup properly,
we should always have at least one TPU VM set up to receive requests, and this
can scale up up to 4 replicas.

You should observe this type of behavior either within the autoscaler logs:
```

(autoscaler +43m15s) Adding 1 node(s) of type ray_tpu.
```

or within the Ray Serve logs:

```
...
---------------------------------------------------------------
Usage:
 0.0/240.0 CPU
 1.0/1.0 TPU (1.0 used of 1.0 reserved in placement groups)
 0B/287.66GiB memory
 0B/123.87GiB object_store_memory

Demands:
 {'TPU': 1.0} * 1 (PACK): 99+ pending placement groups
2023-08-25 16:17:20,553	INFO autoscaler.py:1374 -- StandardAutoscaler: Queue 1 new nodes for launch
2023-08-25 16:17:20,553	INFO autoscaler.py:470 -- The autoscaler took 0.191 seconds to complete the update iteration.
2023-08-25 16:17:20,553	INFO node_launcher.py:166 -- NodeLauncher1: Got 1 nodes to launch.
```

In case there is a lower amount of demand, then RayServe will autoscale down:

```
(autoscaler +1h8m33s) Removing 1 nodes of type ray_tpu (idle).
(autoscaler +1h8m43s) Resized to 244 CPUs.
```


### Tearing down the cluster
Once you are finished developing, you can tear down your cluster as follows:

```
$ ray down -y cluster.yaml
```

