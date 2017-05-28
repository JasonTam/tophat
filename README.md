# TiefRex

## Syncing Data Fixtures
`data/get_fixture_small.sh` will grab 10 days of interaction data and some related dimensional data
 
## Running Local
*Only* local runs are supported right now.

`python tiefrex/fit_job.py`

Tensorboard can be run for inspection

`tensorboard --logdir=/tmp/tensorboard-logs`