# TPU Retry script

## About
This is a toy example of a script that can be used to poll for TPU health and delete/create if stuck in an unhealthy state.

Please continue reading for expectations:
- Cloud TPUs are expected to undergo maintenance events, but is expected to recover.
- Generally, saving checkpoints more often (at least every hour) allows you to gracefully recover. Also ensure that your training script resumes from checkpoint correctly.
- In unexpected circumstances, it's possible that the TPU does not recover from maintenance event. This script showcases an example of how to detect this and delete/re-create the TPU.
- If your process running the training script crashes, you can modify this script to re-try running the train script.


## Example usage
You can run this script on your VM. This assumes that you have already exported the tpu name, e.g.
```
export TPU_NAME={my_tpu}
```

Within your VM you can run this script:
```
./retry.sh $TPU_NAME &
```

You are free to modify the polling frequency, the re-creation logic, etc.

Note that this runs indefinitely, so you will need to `pkill` the script once you are done.

*NOTE*: Please modify the script to reflect the correct TPU deletion/creation command, as this will differ if you're using e.g. TPU VM or a reserved TPU.
