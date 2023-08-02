
import numpy as np
import pandas as pd

import re
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

import pickle
warnings.simplefilter('ignore')

data = pd.read_csv('Language Detection.csv')
print("Dataset Loaded...")

# Engineering
X, y = data['Text'], data['Language']

y =  LabelEncoder().fit_transform(y)

# mapping
data_list = []

for text in X:
    # we want to ignore figures and alphanumerics
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    data_list.append(text.lower())

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# creating bags of words
cv = CountVectorizer().fit(X_train)
X_train = cv.transform(X_train).toarray()
X_test = cv.transform(X_test).toarray()
print("Data Preprocessing and engineering successful")
print("Training a Multinomial Naive Baye Model ...")
#training
model = MultinomialNB()
model.fit(X_train, y_train)

# testing
preds = model.predict(X_test)
acc = accuracy_score(preds, y_test)
print('First Iteration: Accuracy score = ', acc)

# pipeline
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.20, random_state=42)
pipeline = Pipeline([('vectorizer', cv), ('multinomialNB', model)])
#pipeline.fit(X_train2, y_train2)

preds2 = pipeline.predict(X_test2)
acc2 = accuracy_score(preds2, y_test2)
print('Second Iteration with Pipeline: Accuracy Score', acc2)
print("Training, Testing and pipeling done...")

# exporting model
MODEL_VERSION = '0.1.0'
model_path = f'trained_pipeline-{MODEL_VERSION}.pkl'

with open(model_path, 'wb') as file:
    pickle.dump(pipeline, file)
print(f'Model successfully saved as {model_path} on storage')