import pickle
import re
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


labels = ['obscene', 'insult', 'toxic',
          'severe_toxic', 'identity_hate', 'threat']

train_df = pd.read_csv('./data/train.csv')

train_df = train_df.drop('char_length', axis=1)

X = train_df.comment_text

vect = TfidfVectorizer(max_features=5000, stop_words='english')

X_dtm = vect.fit_transform(X)


logreg = LogisticRegression(C=12.0)

for label in labels:
    print('... Processing {}'.format(label))
    y = train_df[label]

    # open a file, where you ant to store the data
    file = open(label+'.pickle', 'wb')

    # train the model using X_dtm & y
    logreg.fit(X_dtm, y)

    # dump model to the specified file
    pickle.dump(logreg, file)

    # close the file
    file.close()
