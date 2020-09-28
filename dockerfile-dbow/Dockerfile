FROM python:2
MAINTAINER honsiongchs@gmail.com

# Fix for numpy
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

# Fix for scipy
RUN apt-get update && apt-get install -y gfortran libopenblas-dev liblapack-dev

COPY requirements.txt ./
RUN pip install -r requirements.txt

ADD keras.json /root/.keras/keras.json
ADD theanorc /root/.theanorc
COPY . .

EXPOSE 6002
CMD ["python", "dbow.py"]
