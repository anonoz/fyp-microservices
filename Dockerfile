FROM python:2-onbuild
MAINTAINER honsiongchs@gmail.com

ADD keras.json /root/.keras/keras.json
ADD theanorc /root/.theanorc
COPY . /app/

EXPOSE 6001 6002
