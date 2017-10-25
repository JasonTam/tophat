FROM python:3.6

RUN mkdir /tmp/tensorboard-logs

RUN pip install --upgrade awscli

WORKDIR /opt/cerebro-deep-rec-engine

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install tensorflow==1.3.0

ADD tiefrex /opt/cerebro-deep-rec-engine/tiefrex
ENV PYTHONPATH=$PYTHONPATH:/opt/cerebro-deep-rec-engine/tiefrex

ADD bin/run.sh bin/run.sh

ENTRYPOINT ["bin/run.sh"]
