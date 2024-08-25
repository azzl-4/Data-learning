#%% 導入需要套件
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

#%% 下載 nltk 資源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

#%% 導入數據集 
file_path = "Suicide_Detection.csv"
df = pd.read_csv(file_path)

#%% 替換suicide與non-suicide的標籤<label>
df['label'] = df['class'].map({'suicide': 1, 'non-suicide': 0})

####清理文本####

#%% 定義變數
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
custom_stopwords = set(['i', 'a', 'u', 'im', 'said', 'ex'])

#%% Step 1: 清理文本函數 >>跑很久
def clean_and_filter_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]|http\S+|[^a-zA-Z\s]', ' ', text)
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in stop_words and word not in punctuation and word not in custom_stopwords])

df['cleaned_text'] = df['text'].apply(clean_and_filter_text)

#%% Step 2: 提取形容詞方便進行視覺化
def extract_adjectives(text):
    words = word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    return [word for word, tag in tagged_words if tag in ('JJ', 'JJR', 'JJS')]

suicide_texts = df[df['label'] == 1]['cleaned_text']
non_suicide_texts = df[df['label'] == 0]['cleaned_text']

#%% 擷取兩個類別的形容詞
suicide_adjectives = [adj for text in suicide_texts for adj in extract_adjectives(text)]
non_suicide_adjectives = [adj for text in non_suicide_texts for adj in extract_adjectives(text)]

#%% Step 3: 頻率分佈和可視化
def plot_top_adjectives(adjectives, title, ax):
    freq_dist = nltk.FreqDist(adjectives)
    most_common = freq_dist.most_common(10)
    words, counts = zip(*most_common)
    
    sns.barplot(x=list(counts), y=list(words), ax=ax, palette='Blues_d')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Frequency', fontsize=14)
    ax.set_ylabel('Adjective', fontsize=14)
    
fig, axes = plt.subplots(nrows=2, figsize=(10, 10))
plot_top_adjectives(suicide_adjectives, "Top 10 Adjectives in Suicide Texts", axes[0])
plot_top_adjectives(non_suicide_adjectives, "Top 10 Adjectives in Non-Suicide Texts", axes[1])

plt.tight_layout()
plt.show()

#%% Step 4: 自殺高頻前25個情感形容詞的文字雲
def create_wordcloud(filtered_words):
    word_freq = nltk.FreqDist(filtered_words)
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, background_color="gray", colormap="Blues").generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

create_wordcloud(suicide_adjectives)

#%% Step 5: 模型管道 - 更新特徵工程
vectorizer = CountVectorizer(max_features=10000, min_df=5, max_df=0.9)
X_text_features = vectorizer.fit_transform(df['cleaned_text'])


#來自 nltk.sentiment.vader 模塊。SentimentIntensityAnalyzer 是用於進行情緒分析的工具，它可以對文本進行情感強度的評估，並返回文本的正面、負面、中性及綜合情感分數先将稀疏矩阵 X_text_features 转换为密集矩阵（使用 .toarray()），然后将其与情绪得分（pos, neu, neg, compound）结合在一起，最终形成一个完全密集的 X 特征矩阵。
#文本的特徵向量與情緒分數進行合併，生成最終的特徵矩陣 X，矩陣將用於機器學習模型的訓練和測試`,`適合數據集小的data

#sia = SentimentIntensityAnalyzer()
#將結果存儲到數據框 df
#df[['pos', 'neu', 'neg', 'compound']] = df['cleaned_text'].apply(lambda x: pd.Series(sia.polarity_scores(x)))

#X = pd.concat([pd.DataFrame(X_text_features.toarray()), df[['pos', 'neu', 'neg', 'compound']]], axis=1)
#y = df['label'] #1 表示自殺，0 表示非自殺

#%%  結合情緒分數
from scipy.sparse import hstack

# 假设 sentiment_scores 是情绪得分的 DataFrame，保持X_text_features为稀疏矩阵，不使用 .toarray()
sentiment_scores = df[['pos', 'neu', 'neg', 'compound']].values  # 这部分通常是密集的
# 使用 scipy.sparse.hstack 将稀疏矩阵与密集矩阵结合
X = hstack([X_text_features, sentiment_scores])
y = df['label'] #1 表示自殺，0 表示非自殺

#%% 分層分割和交叉驗證
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#model = MultinomialNB()
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cross_val_scores.mean() * 100:.2f}%")


#%% 訓練和評估模型
# 使用 Logistic 回歸代替 MultinomialNB
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))


#%% 美化混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix", fontsize=16)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.show()

#%% 優化預測功能 - 依據輸入文本返回自殺概率
def predict_suicide_probability(text):
    cleaned_text = clean_and_filter_text(text)
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    
    pos, neu, neg, compound = pd.Series(sia.polarity_scores(cleaned_text)).values
    combined_features = pd.concat([pd.DataFrame(text_vector), pd.DataFrame([[pos, neu, neg, compound]])], axis=1)
    
    suicide_probability = model.predict_proba(combined_features)[:, 1][0]
    return f"Suicide Probability: {suicide_probability * 100:.2f}%"

#%% 測試輸入
test_text = "I feel hopeless and can't see a future."
print(predict_suicide_probability(test_text))
# %%
