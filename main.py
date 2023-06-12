import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# テストデータの準備
test_text = ["The movie is good!"]
max_len = 500  # 文章の最大長さ

# テキストデータの前処理
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(test_text)
test_sequences = tokenizer.texts_to_sequences(test_text)
test_sequences = pad_sequences(test_sequences, maxlen=max_len)

# 保存したモデルの読み込み
loaded_model = load_model('sentiment_model.h5')

# テストデータの分類（保存したモデルを使用）
prediction = loaded_model.predict(test_sequences)

# 分類結果の表示
if prediction[0] > 0.5:
    print(f"Positive sentiment {prediction[0]}")
else:
    print(f"Negative sentiment {1-prediction[0]}")

# 結果をCSVファイルに保存
# result = pd.DataFrame({'Text': test_text, 'Sentiment': prediction.squeeze()})
# result.to_csv('sentiment_results.csv', index=False)