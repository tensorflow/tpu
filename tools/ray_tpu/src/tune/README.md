# RayTune on Cloud TPUs
We provide an example that showcases how to tune a Flax model on TPU VMs
using [RayTune](https://docs.ray.io/en/latest/tune/index.html).

## How it works

`run_hp_search.py` defines a toy tuning example that uses an MNIST training
example with Flax to tune across the `momentum` parameter.

## Starting your Ray cluster
Before starting, make sure you change your `project_id` within `cluster.yaml`
to your GCP project and that your GCP project has (1) the TPU API enabled, and
(2) proper TPU quotas granted.

Navigate to this folder:

```
$ cd src/tune
```

and make sure you have the requirements installed:

```
$ pip3 install -r requirements.txt
```

Then start your Ray cluster as follows:

```
$ ray up -y cluster.yaml
Cluster: ray-tune-flax

Checking GCP environment settings
...

2023-08-25 15:54:24,083	INFO node.py:311 -- wait_for_compute_zone_operation: Waiting for operation operation-1692978863799-603c15bc9fcd0-16e91745-2eb95a63 to finish...
2023-08-25 15:54:29,257	INFO node.py:330 -- wait_for_compute_zone_operation: Operation operation-1692978863799-603c15bc9fcd0-16e91745-2eb95a63 finished.
  New status: up-to-date

Useful commands
  Monitor autoscaling with
    ray exec /home/$USER/src/tune/cluster.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
  Connect to a terminal on the cluster head:
    ray attach /home/$USER/src/tune/cluster.yaml
  Get a remote shell to the cluster manually:

```

### Pulling up the Ray Dashboard
Once the Ray cluster is up, you can connect to the Ray dashboard with the following command:

```
$ ray dashboard cluster.yaml 
...
2023-07-10 16:19:24,064	INFO log_timer.py:25 -- NodeUpdater: ray-ray-flax-tune-head-523354b9-compute: Got IP  [LogTimer=0ms]
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
2023-08-25 15:48:02,455	INFO autoscaler.py:594 -- StandardAutoscaler: Terminating the node with id projects/googles-secret-dev-project/locations/us-central2-b/nodes/ray-ray-tune-flax-worker-b9a8d2bc-tpu and ip 10.130.0.91. (outdated)
2023-08-25 15:48:02,456	INFO node_provider.py:186 -- NodeProvider: projects/googles-secret-dev-project/locations/us-central2-b/nodes/ray-ray-tune-flax-worker-b9a8d2bc-tpu: Terminating node
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

To get information about the Ray cluster, you can also connect to an interactive
environment on the Ray head node with

```
$ ray attach cluster.yaml
```

From there, you can poll the status of the Ray cluster:

```
$ ubuntu@ray-ray-flax-tune-head-523354b9-compute:~$ ray status
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
Set RAY_HEAD_IP=10.130.0.158
Set RAY_ADDRESS=http://10.130.0.158:8265
```

### Running the Tune Job
Once your Ray cluster is up and running and `RAY_ADDRESS` is set, you can trigger the tuning job:

```
$ ray job submit --working-dir . -- python run_hp_search.py
Job submission server address: http://10.130.0.158:8265
2023-08-25 16:17:08,103	INFO dashboard_sdk.py:338 -- Uploading package gcs://_ray_pkg_561bb5a079855829.zip.
2023-08-25 16:17:08,104	INFO packaging.py:520 -- Creating a file package for local directory '.'.

-------------------------------------------------------
Job 'raysubmit_fAZRRVStVyyS5xnu' submitted successfully
-------------------------------------------------------

Next steps
  Query the logs of the job:
    ray job logs raysubmit_fAZRRVStVyyS5xnu
  Query the status of the job:
    ray job status raysubmit_fAZRRVStVyyS5xnu
  Request the job to be stopped:
    ray job stop raysubmit_fAZRRVStVyyS5xnu

Tailing logs until the job exits (disable with --no-wait):
...
== Status ==
Current time: 2023-08-25 16:17:15 (running for 00:00:00.85)
Using FIFO scheduling algorithm.
Logical resource usage: 0/240 CPUs, 0/0 GPUs (0.0/1.0 TPU)
Result logdir: /home/ubuntu/ray_results/hp_search_mnist_2023-08-25_16-17-14
Number of trials: 100/100 (100 PENDING)
+-----------------------------+----------+-------+------------+
| Trial name                  | status   | loc   |   momentum |
|-----------------------------+----------+-------+------------|
| hp_search_mnist_da45f_00000 | PENDING  |       |   0.212777 |
| hp_search_mnist_da45f_00001 | PENDING  |       |   0.721762 |
| hp_search_mnist_da45f_00002 | PENDING  |       |   0.268339 |
| hp_search_mnist_da45f_00003 | PENDING  |       |   0.113011 |
| hp_search_mnist_da45f_00004 | PENDING  |       |   0.208454 |
...
```

As the demo runs, you will see results populate:

```
...
Result logdir: /home/ubuntu/ray_results/hp_search_mnist_2023-08-25_16-17-14
Number of trials: 100/100 (99 PENDING, 1 RUNNING)
+-----------------------------+----------+--------------------+------------+--------+--------+------------------+
| Trial name                  | status   | loc                |   momentum |    acc |   iter |   total time (s) |
|-----------------------------+----------+--------------------+------------+--------+--------+------------------|
| hp_search_mnist_da45f_00000 | RUNNING  | 10.130.0.159:14536 |   0.212777 | 0.9896 |      2 |          55.5939 |
...
```

If you observe the autoscaling logs (`ray monitor cluster.yaml`) you should also
see that the Ray Autoscaler triggers:

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
2023-08-25 16:17:20,553	INFO autoscaler.py:1374 -- StandardAutoscaler: Queue 4 new nodes for launch
2023-08-25 16:17:20,553	INFO autoscaler.py:470 -- The autoscaler took 0.191 seconds to complete the update iteration.
2023-08-25 16:17:20,553	INFO node_launcher.py:166 -- NodeLauncher1: Got 4 nodes to launch.
...
```

You will also see autoscaling taking place within the job logs:

```
== Status ==
Current time: 2023-08-25 16:25:05 (running for 00:07:50.72)
Using FIFO scheduling algorithm.
Logical resource usage: 0/240 CPUs, 0/0 GPUs (4.0/1.0 TPU)
Current best trial: da45f_00000 with mean_accuracy=0.9896000027656555 and parameters={'learning_rate': 2.398771478763208e-09, 'momentum': 0.21277710118689364}
Result logdir: /home/ubuntu/ray_results/hp_search_mnist_2023-08-25_16-17-14
Number of trials: 100/100 (94 PENDING, 4 RUNNING, 2 TERMINATED)
...
```

### Stopping the Tune Job
If at any point you want to stop the tuning job before it runs to completion, you
can easily do so by stopping the associated job.

When starting the job, Ray will tell you the job ID, but in case you lost it
you can also poll ray for that information:

```
$ ray job list
Job submission server address: http://10.130.0.158:8265
[JobDetails(type=<JobType.SUBMISSION: 'SUBMISSION'>, job_id='02000000', submission_id='raysubmit_fAZRRVStVyyS5xnu', driver_info=DriverInfo(id='02000000', node_ip_address='10.130.0.158', pid='10377'), status=<JobStatus.RUNNING: 'RUNNING'>, entrypoint='python run_hp_search.py', message='Job is currently running.', error_type=None, start_time=1692980228147, end_time=None, metadata={}, runtime_env={'working_dir': 'gcs://_ray_pkg_561bb5a079855829.zip'}, driver_agent_http_address='http://10.130.0.158:52365', driver_node_id='63f83d9d8b88068e73a4662bd757bed0c5aaf618fdb4d7f4d18f9310')]
...
$ ray job stop raysubmit_fAZRRVStVyyS5xnu
Job submission server address: http://10.130.0.158:8265
Attempting to stop job 'raysubmit_fAZRRVStVyyS5xnu'
Waiting for job 'raysubmit_fAZRRVStVyyS5xnu' to exit (disable with --no-wait):
Job has not exited yet. Status: RUNNING
Job has not exited yet. Status: RUNNING
Job has not exited yet. Status: RUNNING
Job 'raysubmit_fAZRRVStVyyS5xnu' was stopped
```

### Tearing down the cluster
Once you are finished developing, you can tear down your cluster as follows:

```
$ ray down -y cluster.yaml
```
