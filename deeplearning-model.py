from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


train = pd.read_csv('./data/train.csv')

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene",
                 "threat", "insult", "identity_hate"]].values

max_features = 30000
maxlen = 100
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)

# model
inp = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(GRU(100, return_sequences=True,
                      dropout=0.2, recurrent_dropout=0.2))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])
outp = Dense(6, activation="sigmoid")(conc)

model = Model(inputs=inp, outputs=outp)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# hyperparameters
batch_size = 128
epochs = 6

# positive labels per category - to keep a balanced train and validation label category distribution split
pos_labels_dist = np.sum(y_train, 1)
X_tra, X_val, y_tra, y_val = train_test_split(
    x_train, y_train, stratify=pos_labels_dist, train_size=0.75, random_state=42)

# checkpoint
filepath = "weights-improvement-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_loss', save_best_only=True)
callbacks_list = [checkpoint]

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=callbacks_list)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
