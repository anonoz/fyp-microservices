FROM python:2
MAINTAINER honsiongchs@gmail.com

COPY requirements.txt ./
RUN pip install -r requirements.txt

ADD keras.json /root/.keras/keras.json
ADD theanorc /root/.theanorc
COPY . .

EXPOSE 6002
CMD ["python", "dbow.py"]
