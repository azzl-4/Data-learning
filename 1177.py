import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 讀取 CSV 檔案
file_path = 'Suicide_Detection.csv'
df = pd.read_csv(file_path)

# 定義情感分類函數
def get_sentiment(text):
    analysis = TextBlob(text).sentiment.polarity
    if analysis > 0:
        return 'positive'
    elif analysis == 0:
        return 'neutral'
    else:
        return 'negative'

# 對所有文本進行情感分析
df['sentiment'] = df['text'].apply(get_sentiment)

# 將情感分類與原始文本一起作為特徵進行建模
df['combined_features'] = df['text'] + " " + df['sentiment']

# 分割訓練集和測試集
X = df['combined_features']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用TF-IDF進行特徵提取
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 訓練隨機森林模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train_tfidf, y_train)

# 預測
y_pred = model.predict(X_test_tfidf)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['non-suicide', 'suicide'])

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
