"""
skipgram.py contains classifier models that use the pre-trained skipgram vector as feature extractor.

Classifiers:
- LSTM
- GRU
- CNN

Requires 1GB of working memory.
"""
from flask import Flask, request, jsonify, render_template
from random import random
  
# The prediction function
#
# POST /predict
#
def predict_sentiment(classifier, text):
  if classifier not in ['ffnn', 'svm', 'rf']:
    return jsonify(success = False, error = "classifier_not_found")

  fake_positive_score = random()
  prediction = [fake_positive_score, 1 - fake_positive_score]
  polarity = 'positive' if prediction[0] > prediction[1] else 'negative'

  return_json = {
    'success': True,
    'polarity': polarity,  
    'score': prediction,
  }

  return jsonify(**return_json)

# Load REST API
app = Flask(__name__)

print "Loading REST API..."

@app.route('/')
def hello_world():
  return "API only."

@app.route('/predict', methods=['POST'])
def predict():
  request_json = request.get_json(force=True)
  return predict_sentiment(request_json['classifier'], request_json['review'])

if __name__ == "__main__":
  app.run(host='0.0.0.0', port='6002')
