import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 讀取 CSV 檔案
file_path = 'Suicide_Detection.csv'
df = pd.read_csv(file_path)

# 篩選出標記為 'suicide' 的文本
suicide_texts = df[df['class'] == 'suicide']['text']

# 文本清理函數
def clean_text(text):
    text = text.lower()  # 轉換為小寫
    text = re.sub(r'\[.*?\]', '', text)  # 移除括號內容
    text = re.sub(r'http\S+', '', text)  # 移除網址
    text = re.sub(r'\W', ' ', text)  # 移除非字母字符
    text = re.sub(r'\s+', ' ', text)  # 移除多餘空格
    return text

# 對所有文本進行清理
cleaned_texts = suicide_texts.apply(clean_text)

# 使用 CountVectorizer 來計算詞頻
vectorizer = CountVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(cleaned_texts)

# 取得高頻詞彙和它們的出現次數
freq_words = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
freq_words = sorted(freq_words, key=lambda x: x[1], reverse=True)

# 生成詞雲
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(freq_words))

# 顯示詞雲
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# 列出前20個高頻詞彙
print("Top 20 frequent words in 'suicide' texts:")
for word, freq in freq_words[:20]:
    print(f"{word}: {freq}")
