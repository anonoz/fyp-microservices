"""
skipgram.py contains classifier models that use the pre-trained skipgram vector as feature extractor.

Classifiers:
- LSTM
- GRU
- CNN

Requires 1GB of working memory.
"""
from flask import Flask, request, jsonify, render_template
from text_tokenizer import tokenize
from gensim.models import Word2Vec
from keras.models import load_model
import sys
import numpy as np

word_vector_bin_file = "wordvectors/skipgram-100d.bin"
cnn_model_file = "models/cnn-skipgram-100d.h5"
lstm_model_file = "models/lstm-skipgram-100d.h5"
gru_model_file = "models/gru-skipgram-100d.h5"

# Load word vectors
print "Loading word2vec..."
word_vecs = Word2Vec.load_word2vec_format(word_vector_bin_file, binary=True)
word_vecs_dims = word_vecs.syn0.shape[1]

print "Loading CNN..."
cnn_model = load_model(cnn_model_file)
cnn_batch_input_shape = cnn_model.get_config()[0]['config']['layers'][0]['config']['batch_input_shape']
cnn_max_review_length = cnn_batch_input_shape[1]

print "Loading LSTM..."
lstm_model = load_model(lstm_model_file)

print "Loading GRU..."
gru_model = load_model(gru_model_file)

def word_vector_for(word):
  """
  Returns a tuple of word vectors and an integer indicating whether the word
  exists in the word vector vocabulary. If not exist, randomly generate the
  vectors arounds 0.
  """
  try:
    return (word_vecs[word], 1)
  except KeyError:
    return (np.random.uniform(-0.25, 0.25, word_vecs_dims), 0)
  
# The prediction function
def predict_sentiment(classifier, text):
  if classifier not in ['cnn', 'lstm', 'gru']:
    return jsonify(success = False, error = "classifier_not_found")

  tokens = tokenize(text)
  max_review_length = cnn_max_review_length if classifier == 'cnn' else len(tokens)
  word_vec_array = np.full(fill_value=0.0,
                           shape=(1, max_review_length, word_vecs_dims),
                           dtype='float32')
  word_vec_hits = []
  for i, word in enumerate(tokens):
    word_vec_array[0][i], word_vec_hit = word_vector_for(word)
    word_vec_hits.append(word_vec_hit)

  keras_model = {
    'cnn':  cnn_model,
    'lstm': lstm_model,
    'gru':  gru_model
  }[classifier]

  prediction = keras_model.predict(word_vec_array, batch_size=1)[0]
  polarity = 'positive' if prediction[0] > prediction[1] else 'negative'
  word_vec_hit_rate = np.mean(word_vec_hits)
  
  return_json = {
    'success': True,
    'polarity': polarity,  
    'score': prediction.tolist(),
    'word2vec_hit_rate': word_vec_hit_rate
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
  app.run(host='0.0.0.0', port='6001')
