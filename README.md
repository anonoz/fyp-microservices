# Sentiment Analysis Classifier Microservices

These microservices expose a REST API for client to POST a sentiment analysis prediction request. Predictions are not CPU/GPU taxing, but storing the word vectors in the memory is.

## Word Vectors



## Starting it

```
$ python skipgram.py
```

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

### Responses
