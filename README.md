# Positive or Negative sentiment
IMDB映画レビューデータセットで学習したモデルを使用して, 与えられた文章が Positive か Negative な sentiment かを分類

## Version
- python 3.11.3  
- tensorflow 2.12.0  
- gradio 3.23.0  

## 使い方（Usage）
- まず, ```model_save.py```を実行してください.  IMDB映画レビューデータセットを用いて model を生成し, 保存 
パラメータや層の設定は各自調整してください.  
デフォルトでは以下のモデル名になっています. 
```
'sentiment_model.h5'
```  

- 次に, ```main.py```を実行することで```main.py```内にある```test_text```が Positive か Negative な sentiment かを分類
デフォルトでは以下の文が格納されています. 
```
test_text = ["This movie is good"]
```
