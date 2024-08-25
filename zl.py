# %% 導入需要套件
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag

# %% 下载所需数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

# %% 導入數據集
file_path = "C:\\Users\\azzl\\OneDrive\\文件\\machine_learning\\Data-learning\\Suicide_Detection.csv"
df = pd.read_csv(file_path)

# %% 查看數據的前幾行
print(df.head())

# %% 替換suicide與non-suicide的標籤<label>
df['label'] = df['class'].map({'suicide': 1, 'non-suicide': 0})

# %% 定義清理文本(沒意義的詞)的函數
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
stopwords_custom = set(['i', 'a', 'u', 'im', 'said', 'ex'])
analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_words(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words and word not in punctuation and word not in stopwords_custom]
    return ' '.join(filtered_words)

# %% 應用清理文本和過濾停用詞函數
df['cleaned_text'] = df['text'].apply(lambda x: filter_words(clean_text(x)))

# %% 定義副詞與形容詞提取函數
def extract_adjectives_adverbs(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    adjectives = [word for word, tag in tagged_words if tag in ('JJ', 'JJR', 'JJS')]
    adverbs = [word for word, tag in tagged_words if tag in ('RB', 'RBR', 'RBS')]
    return adjectives, adverbs
#%%
def get_sentiment_scores(text):
    scores = analyzer.polarity_scores(text)
    return scores['pos'], scores['neu'], scores['neg'], scores['compound']

def calculate_weighted_emotion(text):
    adjectives, adverbs = extract_adjectives_adverbs(text)
    pos_sum, neu_sum, neg_sum, compound_sum = 0, 0, 0, 0
    for word in adjectives + adverbs:
        word_score = analyzer.polarity_scores(word)
        pos_sum += word_score['pos']
        neu_sum += word_score['neu']
        neg_sum += word_score['neg']
        compound_sum += word_score['compound']
    return pos_sum, neu_sum, neg_sum, compound_sum

# %% 計算詞頻特徵
vectorizer = CountVectorizer(max_features=10000, min_df=5, max_df=0.9)
X_text_features = vectorizer.fit_transform(df['cleaned_text'])

# %% 計算VADER情感得分
df[['pos', 'neu', 'neg', 'compound']] = df['cleaned_text'].apply(lambda x: pd.Series(get_sentiment_scores(x)))

# %% 定義生成詞雲的函數
def create_wordcloud(words, title):
    word_freq = nltk.FreqDist(words)
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30]
    wordcloud_dict = {word: freq for word, freq in top_words}
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate_from_frequencies(wordcloud_dict)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# %% 生成形容詞和副詞的詞雲
all_adjectives_adverbs = []
for text in df['cleaned_text']:
    adjectives, adverbs = extract_adjectives_adverbs(text)
    all_adjectives_adverbs.extend(adjectives + adverbs)

filtered_words = [word for word in all_adjectives_adverbs if word not in stop_words and word not in punctuation]
create_wordcloud(filtered_words, 'Adjectives and Adverbs Wordcloud')

# %% 生成所有文本的詞雲
all_words = ' '.join([text for text in df['cleaned_text']])
create_wordcloud(all_words.split(), 'All Text Wordcloud')


# %% 定义生成词云的函数
def create_wordcloud(words, title, ax):
    word_freq = nltk.FreqDist(words)
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30]
    wordcloud_dict = {word: freq for word, freq in top_words}
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate_from_frequencies(wordcloud_dict)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)

# %% 创建图形和子图
fig, axs = plt.subplots(1, 3, figsize=(30, 10))

# 生成形容词和副词的词云
all_adjectives_adverbs = []
for text in df['cleaned_text']:
    adjectives, adverbs = extract_adjectives_adverbs(text)
    all_adjectives_adverbs.extend(adjectives + adverbs)
filtered_words = [word for word in all_adjectives_adverbs if word not in stop_words and word not in punctuation]
create_wordcloud(filtered_words, 'Adjectives and Adverbs Wordcloud', axs[0])
#%%
# 生成所有文本的词云
all_words = ' '.join([text for text in df['cleaned_text']])
create_wordcloud(all_words.split(), 'All Text Wordcloud', axs[1])
#%%
# 生成词云函数的词云
create_wordcloud(['create', 'wordcloud', 'function'], 'Wordcloud Function', axs[2])

# 显示图形
plt.tight_layout()
plt.show()





# %% 組合特徵
X = pd.concat([pd.DataFrame(X_text_features.toarray()), df[['pos', 'neu', 'neg', 'compound']]], axis=1)
y = df['label']

# %% 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% 交叉驗證
model = MultinomialNB()
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"交叉驗證平均準確率: {cross_val_scores.mean() * 100:.2f}%")

# %% 訓練模型
model.fit(X_train, y_train)

# %% 預測
y_pred = model.predict(X_test)

# %% 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print(f"測試集準確率: {accuracy * 100:.2f}%")

# %% 混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("混淆矩陣")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% 分類報告
print(classification_report(y_test, y_pred))

# %% 預測自殺機率
def predict_suicide_probability(text):
    cleaned_text = clean_text(text)
    filtered_text = filter_words(cleaned_text)
    text_vector = vectorizer.transform([filtered_text]).toarray()
    pos, neu, neg, compound = get_sentiment_scores(filtered_text)
    pos_weighted, neu_weighted, neg_weighted, compound_weighted = calculate_weighted_emotion(text)
    combined_features = pd.concat([
        pd.DataFrame(text_vector), 
        pd.DataFrame([[pos, neu, neg, compound, pos_weighted, neu_weighted, neg_weighted, compound_weighted]])
    ], axis=1)
    suicide_probability = model.predict_proba(combined_features)[:, 1][0]
    return f"該文本的自殺機率為: {suicide_probability * 100:.2f}%"

# %% 測試輸入
test_text = "I feel hopeless and can't see a future."
print(predict_suicide_probability(test_text))
