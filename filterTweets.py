'''
Adapted from 



'''
import pandas as pd
import re
import numpy as np

from keras import models,preprocessing
from keras.layers import  Dense, Dropout, Embedding, LSTM
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import gensim
from nltk import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


'''
Using the new Sentiment140 data, the columns are specified as follows
Target of sentiment, tweet Id, date of tweet, Flag (not important for us), User, Text of the tweet
'''
INPUT_FILE = "./input/training.1600000.processed.noemoticon.csv"
COLUMNS = ["SENTIMENT", "ID", "DATE", "FLAG", "USER", "TEXT"]
TRAIN_COLUMNS = ["SENTIMENT", "TEXT"]

'''
We need to scrub the text of useless characters and show just the important parts.
In almost every project there are symbols that are agreed to be useless/counteracting.
These are the symbols that we wont allow in the text.

NOTE: I found that removing stopwords actually decreases the accuracy of predictions so I didn't
'''
# removes links, @s,values, and non-alphabet characters
pattern = "@\S+|https?:\S+|http?:\S|[^A-Za-z]+"
# stemmer for removing unnecessary characters like "Played -> play"
stemmer = SnowballStemmer(language="english")

def scrubText(text, stem=False):
    # substitute all chars in pattern with space and lowercase them
    text = re.sub(pattern, ' ', str(text).lower()).strip()
    tokens = []
    # split the text into words 
    for token in text.split():
        # if it exists in stem, cut it down to essentials
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    # return the text in the tokenized and scrubbed format
    return " ".join(tokens)


'''
I will vectorize the words by training and calling the Word2Vec model on the corpus I created.
Typically on larger datasets, SkipGram (sg = 1) is better and thus I will use that.
This will give us the words as vectors which allows for mathematical equations on the words.
Euclidian distance etc....
'''

def vectorizeTextReturnModel(data_frame_train,custom_corpus):
    # train model on custom corpus
    # min_count = don't count words that appear less than 10 times in 1.2million tweets
    # workers = number of partitions on training
    # window = how far back and forward should it look when looking at a word
    # vector_size = number of dimensions (falls off after 300 but takes longer than default 100)
    # sg = SkippGram method which is generally better for large datasets
    model = gensim.models.word2vec.Word2Vec(custom_corpus, min_count=10, vector_size=300, workers=3, window=5, sg=1)
    return model

'''
Scan in tweets and get them ready for parsing
'''
def scanTweets(input_file,columns):
    return pd.read_csv(input_file,names=columns)

# prints the word sentiments for the values of the data frame target value
def printSentiments(num):
    if num == 0:
        return 'NEG'
    elif num == 4:
        return 'POS'
    elif num == 2:
        return 'NEU'

#scan in tweets to data frame
data_frame = scanTweets(INPUT_FILE,COLUMNS)

# print the sentiments
data_frame['SENTIMENT'] = data_frame['SENTIMENT'].apply(lambda x: printSentiments(x))

# scrub the text of unnecessary characters
data_frame['TEXT'] = data_frame['TEXT'].apply(lambda x: scrubText(x))

# split the data into testing (20%) and training (80%)
data_frame_train, data_frame_test = train_test_split(data_frame, train_size=0.8)

# build the corpus, list of lists of strings of our text
custom_corpus =[text.split() for text in data_frame_train['TEXT']]

#train the model on our corpus and return the model
model = vectorizeTextReturnModel(data_frame_train,custom_corpus)

# tokenize the data
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data_frame_train['TEXT'])
size = len(tokenizer.word_index) + 1

# convert words to numpy arrrays
x_train, x_test = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(data_frame_train['TEXT']),maxlen=300), preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(data_frame_test['TEXT']),maxlen=300)

# encode the sentiment column and transform it into a normal form
encoder = LabelEncoder()
encoder.fit(data_frame_train['SENTIMENT'].tolist())
y_train, y_test = encoder.transform(data_frame_train['SENTIMENT'].tolist()), encoder.transform(data_frame_test['SENTIMENT'].tolist())

# reshape the y_train and y_test to be able to be trained on and normalized
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

# create the embedding layer using the SkippGram word2vec model we created above
#initialize numpy array
embed_npmatrix =  np.zeros((size,300))
# if a token exists in the model, set it equal to the value of the model's vector
for token, i in tokenizer.word_index.items():
    if token in model.wv:
        embed_npmatrix[i] = model.wv[token]

print(embed_npmatrix.shape)

'''
Now I specify the layers of the neural net 

NOTE EXPERIMENT WITH THIS PART RIGHT HERE
'''
embedding_layer = Embedding(size,300,weights=[embed_npmatrix], input_length=300,trainable=False)

sentiment_model = models.Sequential()
sentiment_model.add(embedding_layer)
sentiment_model.add(Dropout(0.5))
sentiment_model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
sentiment_model.add(Dense(1, activation='sigmoid'))

sentiment_model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

history = sentiment_model.fit(x_train, y_train,
                    batch_size=1024,
                    epochs=4,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)

score = sentiment_model.evaluate(x_test, y_test, batch_size=1024)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])
