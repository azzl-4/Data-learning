#%%
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import Parallel, delayed
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import multiprocessing

# 讀取 CSV 檔案
file_path = 'Suicide_Detection.csv'
df = pd.read_csv(file_path)
#%%
# 初始化 VADER 分析器
nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()
#%%
# 使用 VADER 進行情感分析的函數
def get_sentiment_vader(text):
    analysis = vader_analyzer.polarity_scores(text)['compound']
    if analysis > 0:
        return 'positive'
    elif analysis == 0:
        return 'neutral'
    else:
        return 'negative'
#%%
# 並行處理情感分析
num_cores = multiprocessing.cpu_count()
df['sentiment'] = Parallel(n_jobs=num_cores)(delayed(get_sentiment_vader)(text) for text in df['text'])

# 將情感分類與原始文本一起作為特徵進行建模
df['combined_features'] = df['text'] + " " + df['sentiment']

# 分割訓練集和測試集
X = df['combined_features']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# 使用TF-IDF進行特徵提取
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
#%%
# 訓練隨機森林模型，減少樹的數量以加快速度
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_tfidf, y_train)
#%%
# 預測
y_pred = model.predict(X_test_tfidf)
#%%
# 評估模型
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['non-suicide', 'suicide'])

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
#%%