#%% 1. Text Preprocessing
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('wordnet')
nltk.download('omw-1.4')
#%%
# Load the dataset
file_path = 'Suicide_Detection.csv'
df = pd.read_csv(file_path)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
#%%
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
# Function to get sentiment scores
def get_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']  # Compound score
# Clean all texts
df['cleaned_text'] = df['text'].apply(clean_text)
df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment)
print(df['sentiment_score'].head())
#%% 2. Bag of Words Transformation (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

# Compute TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), stop_words=custom_stop_words)
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# Get top words based on TF-IDF
tfidf_scores = zip(tfidf_vectorizer.get_feature_names_out(), X_tfidf.sum(axis=0).tolist()[0])
tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
top_20_words = [word for word, score in tfidf_scores[:100]]

#%% 3. Word Embeddings (Word2Vec and BERT)
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Prepare texts for Word2Vec
texts_for_w2v = [simple_preprocess(text) for text in df['cleaned_text']]
word2vec_model = Word2Vec(sentences=texts_for_w2v, vector_size=100, window=5, min_count=1, workers=4)

# Display word similarities
print("Semantic similarities:")
for word in top_20_words[:20]:
    similar_words = word2vec_model.wv.most_similar(word, topn=5)
    print(f"\nSimilar words to '{word}':")
    for similar_word, similarity in similar_words:
        print(f"{similar_word}: {similarity:.2f}")

# Optional: For BERT embeddings, you would use a library such as `transformers` to obtain BERT embeddings. This step is not shown here for brevity.

#%% 4. Text Feature Engineering
# Additional text feature engineering can be performed here if needed.

#%% 5. Feature Selection
from sklearn.feature_extraction.text import CountVectorizer

# Compute word counts
count_vectorizer = CountVectorizer(vocabulary=top_20_words)
X_count = count_vectorizer.fit_transform(df['cleaned_text'])
freq_counts = zip(count_vectorizer.get_feature_names_out(), X_count.sum(axis=0).tolist()[0])
freq_counts = sorted(freq_counts, key=lambda x: x[1], reverse=True)

#%% 6. Visualization: Word Cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(freq_counts))

# Display word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
