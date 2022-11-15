from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model # Add 'load_model'
from joblib import dump, load # For reading the Tokenizer Pickle
import csv

KERAS_MODEL = "model.h5"
TOKENIZER_MODEL = "tokenizer.pkl"

# KERAS
SEQUENCE_LENGTH = 300

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# Load the model and the tokenizer to make predictions
model = load_model(KERAS_MODEL)
tokenizer = load(TOKENIZER_MODEL)

def decode_sentiment(score, include_neutral=False):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict(text, include_neutral=False):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return label, float(score) 
print(predict("the rain makes me want to die"))

with open('2016_01.csv', 'rt',encoding="utf8") as inp, open('2016_01_sentiment.csv','wt') as out:
    writer = csv.writer(out)
    i = 0
    rows = csv.reader(inp)
    for row in rows:
        i+=1
        pred = predict(row[10])
        writer.writerow(pred)
        print(pred)
    print(i)