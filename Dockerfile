FROM python:3.6

RUN apt-get update

ADD . /home/tophat/
WORKDIR /home/

RUN cd tophat && pip install -e .[tf]

