#%%
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import re
import string
from gensim.models import Word2Vec
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# 載入傳統模型及其他資源
knn_model = joblib.load('best_knn_model.pkl')
svm_model1 = joblib.load('best_svm_model.pkl')
svm_model2 = joblib.load('svm_model.joblib')
logistic_regression_model = joblib.load('logistic_regression_model.joblib')
word2vec_model = joblib.load('word2vec_model.pkl')
tfidf_dict = joblib.load('tfidf_dict.pkl')
vectorizer = joblib.load('vectorizer.joblib')

# 載入 RoBERTa 模型和分詞器
roberta_model_path = './suicide_detection_roberta_model'  # 確保這個路徑是正確的
roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path)
roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path)

# 使用 GPU (cuda) 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 將 RoBERTa 模型移動到正確的設備
roberta_model.to(device)

# 停用詞和標點符號
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# 創建分析類
class SuicideRiskEvaluator:
    def __init__(self):
        pass

    # 將句子轉換為 TF-IDF 加權平均向量
    def sentence_to_vector(self, sentence):
        vectors = []
        weights = []
        for word in sentence:
            if word in word2vec_model.wv:
                vectors.append(word2vec_model.wv[word])
                weights.append(tfidf_dict.get(word, 1.0))
        if len(vectors) == 0:
            return np.zeros(word2vec_model.vector_size)
        return np.average(vectors, axis=0, weights=weights)

    # 使用 KNN 模型進行預測
    def predict_with_knn(self, sentence):
        vec = self.sentence_to_vector(sentence)
        vec = vec.reshape(1, -1)
        prediction = knn_model.predict(vec)
        prob = knn_model.predict_proba(vec)
        return {
            'classification': '自殺' if prediction[0] == 1 else '非自殺',
            'suicide_probability': prob[0][1],
            'non_suicide_probability': prob[0][0]
        }

    # 使用 SVM 模型進行預測（第一種實現方式）
    def predict_with_svm1(self, sentence):
        vec = self.sentence_to_vector(sentence)
        vec = vec.reshape(1, -1)
        prediction = svm_model1.predict(vec)
        return {
            'classification': '自殺' if prediction[0] == 1 else '非自殺'
        }

    # 使用 SVM 模型進行預測（帶情感分析的第二種實現方式）
    def predict_with_svm2(self, sentence):
        cleaned_input = self.clean_and_filter_text(sentence)
        X_text_features = vectorizer.transform([cleaned_input])
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(cleaned_input)
        sentiment_vector = [[sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg'], sentiment_scores['compound']]]
        X_input = hstack((X_text_features, sentiment_vector))
        prediction = svm_model2.predict(X_input)
        return {
            'classification': '自殺' if prediction[0] == 1 else '非自殺'
        }

    # 使用 Logistic Regression 模型進行預測
    def predict_with_logistic(self, sentence):
        cleaned_input = self.clean_and_filter_text(sentence)
        X_text_features = vectorizer.transform([cleaned_input])
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(cleaned_input)
        sentiment_vector = [[sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg'], sentiment_scores['compound']]]
        X_input = hstack((X_text_features, sentiment_vector))
        prediction = logistic_regression_model.predict(X_input)
        return {
            'classification': '自殺' if prediction[0] == 1 else '非自殺'
        }

    # 使用 RoBERTa 模型進行預測
    def predict_with_roberta(self, text):
        encoding = roberta_tokenizer.encode_plus(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 確保 input_ids 和 attention_mask 在正確的設備上
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = roberta_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]  # 返回預測標籤

        return "自殺" if prediction == 1 else "非自殺"

    # 文本清理函數
    def clean_and_filter_text(self, text):
        text = text.lower()
        text = re.sub(r'\[.*?\]|http\S+|[^a-zA-Z\s]', ' ', text)
        words = word_tokenize(text)
        return ' '.join([word for word in words if word not in stop_words and word not in punctuation])

# 初始化預測類
evaluator = SuicideRiskEvaluator()

# 創建GUI窗口
def predict_risk():
    text = entry.get().strip()
    selected_model = model_var.get()
    
    if not text:
        messagebox.showwarning("輸入錯誤", "請輸入一些文本。")
        return

    if selected_model == "KNN":
        result = evaluator.predict_with_knn(text.split())
        messagebox.showinfo("KNN 自殺風險預測", 
                            f"分類: {result['classification']}\n"
                            f"自殺機率: {result['suicide_probability']*100:.2f}%\n"
                            f"非自殺機率: {result['non_suicide_probability']*100:.2f}%")
    elif selected_model == "SVM (chiu)":
        result = evaluator.predict_with_svm1(text.split())
        messagebox.showinfo("SVM 自殺風險預測 (chiu)", 
                            f"分類: {result['classification']}")
    elif selected_model == "SVM (azz)":
        result = evaluator.predict_with_svm2(text)
        messagebox.showinfo("SVM 自殺風險預測 (azz)", 
                            f"分類: {result['classification']}")
    elif selected_model == "邏輯回歸":
        result = evaluator.predict_with_logistic(text)
        messagebox.showinfo("邏輯回歸 自殺風險預測", 
                            f"分類: {result['classification']}")
    elif selected_model == "BERT":
        result = evaluator.predict_with_roberta(text)
        messagebox.showinfo("RoBERTa 自殺風險預測", 
                            f"分類: {result}")
    else:
        messagebox.showwarning("模型錯誤", "請選擇一個模型")

# 創建主窗口
root = tk.Tk()
root.title("自殺風險預測系統")
root.geometry('500x400')

# 配置整體風格
root.configure(bg='#f0f0f0')
font_large = ("Arial", 14)
font_small = ("Arial", 10)

# 創建輸入框標籤
label = tk.Label(root, text="輸入要分析的文本：", font=font_large, bg='#f0f0f0')
label.pack(pady=(20, 10))

# 創建輸入框
entry = tk.Entry(root, width=50, font=font_small)
entry.pack(pady=(0, 20))

# 創建模型選擇框
model_var = tk.StringVfar(value="選擇模型")
model_label = tk.Label(root, text="選擇預測模型：", font=font_large, bg='#f0f0f0')
model_label.pack(pady=10)

model_menu = ttk.Combobox(root, textvariable=model_var, font=font_small)
model_menu['values'] = ("KNN", "SVM (chiu)", "SVM (azz))", "邏輯回歸", "BERT")
model_menu.pack(pady=10)

# 創建預測按鈕
predict_button = tk.Button(root, text="進行預測", font=font_small, bg='#FF9800', fg='white', command=predict_risk)
predict_button.pack(pady=20, ipadx=10, ipady=5)

# 啟動GUI主循環
root.mainloop()

# %%

#I don't see a way out of this pain. I'm contemplating ending my life自殺範例
#I am excited about my upcoming vacation. Things are looking positive非自殺範例