# Multi-host TPU Jupyter Notebook Instruction

The instruction shows how to run Jupyter notebook on multi-host TPU.

These examples are not meant to be used in production services and are for illustrative purposes only.

## Overview

## Set up ray cluster (one-time)

1.Create a CPU admin:
```
# on cloudtop
./create_cpu.sh
```

Note that this scripts installs dependencies on the VM via
[startup script](https://cloud.google.com/compute/docs/instances/startup-scripts/linux)
and automatically blocks until the startup script is complete.

2.Deploy local code to CPU:
```
# on cloudtop
./deploy_to_admin.sh
```

3.SSH to the VM
```
# on cloudtop
gcloud compute ssh $USER-admin -- -L8265:localhost:8265 -L8888:localhost:8888
```
Note that we enable port forwarding here as Ray will automatically start a dashboard at port 8265. From the machine that you SSH to your VM, you will be able to access this dashboard at http://127.0.0.1:8265/. The other port 8888 is for Jupyter Notebook access.

4.Set up your gcloud credentials within the CPU VM:
```
# on CPU VM
gcloud auth login --update-adc
```

5.Run the necessary pip installs:
```
# on CPU VM
pip3 install -r src/requirements.txt
pip3 install -r src/requirements-notebook.txt
```

6.Start the Ray admin:
```
# on CPU VM
ray start --head --port=6379 --resources='{"controller_host": 1}'
```
Note: `--resources='{"controller_host": 1}'` is used to let `ipcontroller` runs on this CPU VM.

## Start Jupyter Notebook
1.Start `ipcontroller` on CPU VM and `ipengine` on each TPU VM host
```
# on CPU VM
python3 src/ipp_tool.py --code_dir=/code/dir/jupyternotebook/may/use \
--tpu_name=$USER-tpu-v4 --tpu_topology=2x2x2 \
--mode=start 
```
Note: the cmd will provision TPU VM if the TPU does not exist. You can find two log lines indicate `ipcontroller` and `ipengine` are started successfully, like
```
I0330 05:36:55.768044 140053141739328 ipp_tool.py:82] ipyparallel controller is started successfully.
I0330 05:41:33.924189 140053141739328 ipp_tool.py:137] ipyparallel engines are started successfully.
```
Within the code directory, two files are generated under `code_dir/ipython/security/` folder: 
  - `ipcontroller-engine.json` is already used in `ipp_tool.py` to start `ipengine` in each TPU host.
  - `ipcontroller-client.json` will be used for client connection in jupyter notebook in step 3.

2.Start Jupyter Notebook
```
# on CPU VM
juypter-lab
```
3.Use the following code block to connect to ipyparallel in the first cell
```
import ipyparallel as ipp
import os
code_dir = '/path/to/code/dir'
rc = ipp.Client(connection_info=os.path.join(code_dir, 'ipython/security/ipcontroller-client.json'))
```
4.Do your development start with this cell magic `%%px --block --group-outputs=engine` in the first line, it will execute your code block on each TPU hosts.

5.Please refer to `jax_example.ipynb` for more details.


## Known Issue
1. When error pops, it does not show the whole stack trace: Note the stack trace is folded in cell output, you need to click it to unfold it. Since it receive the cell output in text, you are not able to click and unfold it.

