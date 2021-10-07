import numpy as np
import tensorflow.python.keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import pickle


# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.python.keras.models import load_model

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = set(load_doc(vocab_filename).split())

# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

	# load all docs in a directory
def process_docs(vocab):
	documents = list()
	doc = load_doc("my_prediction/my_review.txt")
	# clean doc
	tokens = clean_doc(doc, vocab)
	# add to list
	documents.append(tokens)
	return documents

predict_docs = process_docs(vocab)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

encoded_docs = tokenizer.texts_to_sequences(predict_docs)

X = pad_sequences(encoded_docs, maxlen=1317, padding='post')


# load model
model = load_model('my_model.h5')


y=model.predict_classes(np.array(X))


if (y == [[1]]) :
	print("\n good movie \n")
else :
	print("\n not recommended for humans \n")

