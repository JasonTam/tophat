# Tophat
 
## Running Local
*Only* local runs are supported right now.

Grab 10 days of interaction data and some related dimensional data:

```
./get_fixture_small.sh
```

Build and enter Docker container. Run fit job:


```
make cpu
python tophat/fit_job.py
```

or

```
make gpu
python3.6 tophat/fit_job.py
```

Note: [nvidia-docker](github.com/NVIDIA/nvidia-docker) should be installed for 
GPU docker images

Tensorboard can be run for inspection (does not need to be within
docker container as the log dir is shared by a Docker volume).

```
tensorboard --logdir=/tmp/tensorboard-logs
```


## Integration Test
```
make gpu-integration
```