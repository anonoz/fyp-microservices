FROM python:2-onbuild
MAINTAINER honsiongchs@gmail.com

ADD keras.json /root/.keras/keras.json
ADD theanorc /root/.theanorc

EXPOSE 6002
CMD ["python", "dbow.py"]
