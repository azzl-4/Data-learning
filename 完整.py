#%%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
import seaborn as sns
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
nltk.download('wordnet')
nltk.download('omw-1.4')
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import Parallel, delayed
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import multiprocessing

#%%
# Load the dataset
file_path = 'Suicide_Detection.csv'
df = pd.read_csv(file_path)

# Filter texts marked as 'suicide'
suicide_texts = df[df['class'] == 'suicide']['text']

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Extend stop words list
custom_stop_words = list(ENGLISH_STOP_WORDS.union(set(stopwords.words('english'))))
additional_stop_words = ['want', 'feel', 'year', 'like', 'know', 'time', 'really', 'think', 'going', 'thing', 'day', 'make']
custom_stop_words.extend(additional_stop_words)
#%%
# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in custom_stop_words])
    return text
#%%
# Clean all texts
cleaned_texts = suicide_texts.apply(clean_text)
#%%
# Compute TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), stop_words=custom_stop_words)
X_tfidf = tfidf_vectorizer.fit_transform(cleaned_texts)
#%%
# Get top words based on TF-IDF
tfidf_scores = zip(tfidf_vectorizer.get_feature_names_out(), X_tfidf.sum(axis=0).tolist()[0])
tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
top_20_words = [word for word, score in tfidf_scores[:100]]
# %%
# Compute word counts
count_vectorizer = CountVectorizer(vocabulary=top_20_words)
X_count = count_vectorizer.fit_transform(cleaned_texts)
freq_counts = zip(count_vectorizer.get_feature_names_out(), X_count.sum(axis=0).tolist()[0])
freq_counts = sorted(freq_counts, key=lambda x: x[1], reverse=True)
#%%
# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(freq_counts))

#%%
# Display word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#%%
# Display top 20 words with frequencies
print("Top 20 frequent words in 'suicide' texts (by count):")
for word, freq in freq_counts[:20]:
    print(f"{word}: {freq}")

#%%
# Function to generate a list of colors for the gradient
def generate_gradient_colors(n, start_color, end_color):
    """Generate a list of colors transitioning from start_color to end_color."""
    cmap = plt.get_cmap('Blues')  # You can use any colormap or create a custom one
    return [cmap(i / (n - 1)) for i in range(n)]

# Define a color gradient
top_words_df = pd.DataFrame(freq_counts[:20], columns=['Word', 'Frequency'])
plt.figure(figsize=(12, 8))
start_color = '#f7f7f7'  # Light grey
end_color = '#005f73'    # Dark teal
n_colors = len(top_words_df)

# Generate the gradient colors
palette = generate_gradient_colors(n_colors, start_color, end_color)[::-1]  # Reverse the order

# Plot the bar chart with gradient colors
bars = plt.barh(top_words_df['Word'], top_words_df['Frequency'], color=palette)

# Add value labels to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', va='center')

plt.xlabel('Frequency')
plt.title('Top 20 Frequent Words in Suicide Texts')
plt.gca().invert_yaxis()
plt.show()
#%%
# Plot bar chart for top 20 words
top_words_df = pd.DataFrame(freq_counts[:100], columns=['Word', 'Frequency'])
plt.figure(figsize=(12, 8))

#%%
# Prepare texts for Word2Vec
texts_for_w2v = [simple_preprocess(text) for text in cleaned_texts]
model = Word2Vec(sentences=texts_for_w2v, vector_size=100, window=5, min_count=1, workers=4)

# Display word similarities
print("Semantic similarities:")
for word in top_20_words[:20]:
    similar_words = model.wv.most_similar(word, topn=5)
    print(f"\nSimilar words to '{word}':")
    for similar_word, similarity in similar_words:
        print(f"{similar_word}: {similarity:.2f}")
 #%%
 #進行情感分析
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
X_train_tfidf 
X_test_tfidf
 #%%
 
# 訓練隨機森林模型，減少樹的數量以加快速度
model2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model2.fit(X_train_tfidf, y_train)
 #%%
# 預測
y_pred = model2.predict(X_test_tfidf)
 #%%
# 評估模型
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['non-suicide', 'suicide'])

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
 #%%
# 读取数据集
# 设置循环次数
n_iterations = 10
# 存储每次循环的结果
accuracy_results = []
classification_reports = []
for i in range(n_iterations):
    print(f"Iteration {i+1}/{n_iterations}")
    # 分割训练集和测试集
    # # 初始化随机森林分类器
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算准确性并保存
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results.append(accuracy)

    # 计算分类报告并保存
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_reports.append(report)

    print(f"Accuracy: {accuracy:.4f}")

# 输出各次迭代的平均准确性
mean_accuracy = np.mean(accuracy_results)
print(f"\nMean Accuracy over {n_iterations} iterations: {mean_accuracy:.4f}")

# 如果需要，你还可以分析分类报告中的其他指标，如 precision, recall 等。

# %%
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

# 初始化 VADER 分析器
nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()

# 使用 VADER 進行情感分析的函數
def get_sentiment_vader(text):
    analysis = vader_analyzer.polarity_scores(text)['compound']
    if analysis > 0:
        return 'positive'
    elif analysis == 0:
        return 'neutral'
    else:
        return 'negative'

# 並行處理情感分析
num_cores = multiprocessing.cpu_count()
df['sentiment'] = Parallel(n_jobs=num_cores)(delayed(get_sentiment_vader)(text) for text in df['text'])

# 將情感分類與原始文本一起作為特徵進行建模
df['combined_features'] = df['text'] + " " + df['sentiment']

# 初始化用來存儲每次迭代結果的列表
accuracy_results = []
classification_reports = []

# 設置循環次數
n_iterations = 10

for i in range(n_iterations):
    print(f"Iteration {i+1}/{n_iterations}")

    # 分割訓練集和測試集
    X = df['combined_features']
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    # 使用TF-IDF進行特徵提取
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # 訓練隨機森林模型
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # 預測
    y_pred = model.predict(X_test_tfidf)
    
    # 評估模型
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results.append(accuracy)
    
    report = classification_report(y_test, y_pred, target_names=['non-suicide', 'suicide'])
    classification_reports.append(report)
    
    print(f"Model Accuracy in iteration {i+1}: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

# 打印所有迭代的平均結果
mean_accuracy = sum(accuracy_results) / len(accuracy_results)
print(f"\nMean Accuracy over {n_iterations} iterations: {mean_accuracy:.4f}")

# %%
