import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# データの読み込み
max_features = 10000  # 使用する特徴量の数
max_len = 500  # 文章の最大長さ
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# テキストデータの前処理
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts([str(x) for x in x_train])
x_train = tokenizer.texts_to_sequences([str(x) for x in x_train])
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = tokenizer.texts_to_sequences([str(x) for x in x_test])
x_test = pad_sequences(x_test, maxlen=max_len)

# モデルの構築
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_features, 32),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルの学習
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))

# モデルの保存
model.save('sentiment_model.h5')