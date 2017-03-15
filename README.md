# Sentiment Analysis Classifier Microservices

These microservices expose a REST API for client to POST a sentiment analysis prediction request. The training notebooks can be found in <https://github.com/anonoz/fyp-ipynb>.

Script | Classifiers Available
-------|----------------------
`skipgram.py` | <ul><li>`lstm` - Long short-term memory</li><li>`gru` - Gated recurrent unit</li><li>`cnn` - Convolutional neural network</li></ul>
`dbow.py` | <ul><li>`ffnn` - Regular feedforward neural net</li><li>`svm` - Support vector machine</li><li>`rf` - Random forest</li></ul>

## Docker

These microservices have been Dockerized for your convenience. Running the command below will fetch the Docker image from Docker Hub and start the microservice right away, making it available on port 6001/6002 on your machine:

```
# may need sudo to run
$ docker run -p 6001:6001 -it anonoz/fyp-microservices python skipgram.py

# if you want to run both at the same time, open a new terminal and run this
$ docker run -p 6002:6002 -it anonoz/fyp-microservices python dbow.py
```

In case you want to build the Docker image on your own, run:

```
$ docker build -t [image_name] .
```

Then run the commands in the snippet above the above one, replacing `anonoz/fyp-microservices` with your own image name.

## API

```
POST /predict

{
    "classifier": "lstm",
    "review": "This movie is pretty damn good! ..."
}
```

### Parameters

Field      | Description
-----------|--------------------------------------------------
classifier | Can be `lstm`, `gru`, `cnn`, `ffnn`, `svm`, `rf`
review     | The movie review text body to be analyzed.
