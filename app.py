from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


tf.config.run_functions_eagerly(True)

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("./model_lstm.h2")

# Use application context to share the model across requests
app.config['model'] = model

# Classes for classification
classes = ['U.S. NEWS', 'SPORTS AND ENTERTAINMENT', 'PARENTING AND EDUCATION',
       'WORLDNEWS', 'TRAVEL-TOURISM & ART-CULTURE', 'SCIENCE AND TECH',
       'POLITICS', 'ENVIRONMENT', 'GENERAL', 'LIFESTYLE AND WELLNESS',
       'BUSINESS-MONEY', 'MISC', 'EMPOWERED VOICES']






max_length = 150
trunc_type = 'post'
padding_type = 'post'
vocab_size =20000
oov_tok = "<OOV>"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        # Get input data from the request
        data = request.json
        inference_data = data["inference_data"]

        # Make predictions using the loaded model
        pred_class = prediction(inference_data)

        # Return the prediction as JSON
        return jsonify({"prediction": pred_class})

    except Exception as e:
        return jsonify({"error": str(e)})

def prediction(inference_data):

    # Load the tokenizer from the saved file
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)

    # # Fit the tokenizer on your training data
    # tokenizer.fit_on_texts([inference_data])

    # Perform tokenization and padding on the input data
    X = tokenizer.texts_to_sequences([inference_data])
    X = pad_sequences(X, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    print(X)
    # Make predictions using the loaded model
    pred = model.predict(X)
    print(pred)
    pred_value = tf.argmax(pred, axis=1).numpy()
    print(pred_value[0])
    pred_class = classes[pred_value[0]]
    print(pred_class)

    return pred_class

if __name__ == '__main__':
    app.run(debug=True)
