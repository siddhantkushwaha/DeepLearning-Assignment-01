import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

# %%

path = 'A1-Q3_Dataset/mrdata.tsv'

# %%

data = pd.read_csv(path, sep='\t')[['Phrase', 'Sentiment']]

# %%

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2500, lower=True, split=' ')
tokenizer.fit_on_texts(data['Phrase'].values)

X = tokenizer.texts_to_sequences(data['Phrase'].values)
X = tf.keras.preprocessing.sequence.pad_sequences(X)

Y = pd.get_dummies(data['Sentiment'].values).values

# %%

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.20)

# %%

embed_dim = 128
lstm_out = 196
batch_size = 32

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(2500, embed_dim, input_length=X.shape[1]))
model.add(tf.keras.layers.LSTM(lstm_out, dropout=0.5))
model.add(tf.keras.layers.Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# %%

model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=1)

# Sample output ->
# 124848/124848 [==============================] - 1465s 12ms/sample - loss: 1.0198 - acc: 0.5958
# <tensorflow.python.keras.callbacks.History at 0x7f53ade52ef0>

# %%

score, acc = model.evaluate(X_valid, Y_valid, verbose=1, batch_size=batch_size)
print("Score: %.2f" % score)
print("Validation Accuracy: %.2f" % acc)

# Sample output ->
# 31212/31212 [==============================] - 54s 2ms/sample - loss: 0.9431 - acc: 0.6267
# Score: 0.94
# Validation Accuracy: 0.63