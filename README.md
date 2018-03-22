# Tophat
 
 Tophat is a factorization-based recommendation engine built using 
 [TensforFlow](https://www.tensorflow.org/).  
 
 
## Running GILT Models

### Getting Fixture Data
The download scripts `bin/get_fixture_*.sh` can be used to sync some GILT 
fixture data for local testing.

### Fitting a Model
```
python jobs/fit_job.py [env]
```
env can be {`local`, `integ`, `dev`, `prod`}


### Scheduling on Sundial

Production jobs can be scheduled to run using [Sundial](sundial)


## Tensorboard

Tensorboard can be run for inspection (does not need to be within
docker container as the log dir is shared by a Docker volume).

```
tensorboard --logdir=/tmp/tensorboard-logs
```
