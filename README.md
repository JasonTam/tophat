# Tophat

[![Build Status](https://travis-ci.org/saksdirect/tophat.svg?branch=master)](https://travis-ci.org/saksdirect/tophat)
 
 Tophat is a factorization-based recommendation engine built using 
 [TensorFlow](https://www.tensorflow.org/).  
 

## Installation

Installing from PyPi:
```bash
pip install top-hat
```

Installing the master branch from github in development mode, run:
```bash
git clone git@github.com:gilt/tophat.git
cd tophat
pip install -e .
```

Note that by default, installation assumes you already have TensorFlow installed. 
However, if you need, you can include the installation of TensorFlow in the setup extras as following:
(choose the one that's right for you)
```bash
# CPU pypi
pip install top-hat[tf]

# GPU pypi
pip install top-hat[tf_gpu]

# CPU local dev
pip install -e .[tf]

# GPU local dev
pip install -e .[tf_gpu]
```

## Docker Images
There are two provided Dockerfiles: `Dockerfile` and `Dockerfile.gpu`. The latter gpu variant requires [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). 


## Tests

There are some minimal tests in `tests/` which can all be run using `pytest` or `python setup.py test`.


## Related Projects
The initial motivation behind tophat was to port over [LightFM](https://github.com/lyst/lightfm) and [Spotlight](https://github.com/maciejkula/spotlight) into TensorFlow. 

There also are many other [amazing recommender systems out there](https://github.com/grahamjenson/list_of_recommender_systems)
 -- so choose the one that is right for your case.

