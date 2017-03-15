from sklearn import svm
from gensim.models import Doc2Vec
from keras.models import load_model
from bs4 import BeautifulSoup
from text_tokenizer import tokenize
from sklearn.externals import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import re
import sys
import glob

# CONFIG
doc2vec_model_file = "wordvectors/dbow-100d.model"
ffnn_model_file = "models/ffnn-dbow-100d.h5"
svm_clf_file = "models/svm-dbow-100d.pkl"
rf_clf_file = "models/rf-dbow-100d.pkl"

print "Loading doc2vec..."
d2v_model = Doc2Vec.load(doc2vec_model_file)

print "Loading feedforward neural network..."
ffnn = load_model(ffnn_model_file)

print "Loading SVM..."
svm = joblib.load(svm_clf_file)

print "Loading Random Forest..."
rf = joblib.load(rf_clf_file)

# The prediction function
def predict_sentiment(classifier, text):
  if classifier not in ['ffnn', 'svm', 'rf']:
    return jsonify(success=False, error="classifier_not_found")

  tokens = tokenize(text)
  doc_vector = d2v_model.infer_vector(tokens)

  prediction = {
    'svm': lambda doc_vector: svm.predict_proba(doc_vector)[0],
    'rf' : lambda doc_vector: rf.predict_proba(doc_vector)[0],
    'ffnn': lambda doc_vector: ffnn.predict(np.array([doc_vector]), batch_size=1)[0]
  }[classifier](doc_vector)
  polarity = 'positive' if prediction[0] > prediction[1] else 'negative'
  
  return_json = {
    'success': True,
    'polarity': polarity,
    'score': prediction.tolist(),
    'word2vec_hit_rate': 1.00
  }

  return jsonify(**return_json)

# Load REST API
app = Flask(__name__)

print "Loading REST API..."

@app.route('/')
def hello_world():
  return "API only"

@app.route('/predict', methods=['POST'])
def predict_post():
  request_json = request.get_json(force=True)
  return predict_sentiment(request_json['classifier'], request_json['review'])

if __name__ == "__main__":
  app.run(host='0.0.0.0', port='6002')
