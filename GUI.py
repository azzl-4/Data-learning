#%%
import joblib
import tkinter as tk
from tkinter import ttk
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
        prob = svm_model1.predict_proba(vec)  # 確保使用 predict_proba
        return {
            'classification': '自殺' if prediction[0] == 1 else '非自殺',
            'suicide_probability': prob[0][1],
            'non_suicide_probability': prob[0][0]
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
        prob = svm_model2.predict_proba(X_input)  # 確保使用 predict_proba
        return {
            'classification': '自殺' if prediction[0] == 1 else '非自殺',
            'suicide_probability': prob[0][1],
            'non_suicide_probability': prob[0][0]
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
        prob = logistic_regression_model.predict_proba(X_input)  # 確保使用 predict_proba
        return {
            'classification': '自殺' if prediction[0] == 1 else '非自殺',
            'suicide_probability': prob[0][1],
            'non_suicide_probability': prob[0][0]
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
            probs = torch.nn.functional.softmax(logits, dim=1)  # 使用 softmax 轉換為機率
            prediction = torch.argmax(probs, dim=1).cpu().numpy()[0]  # 返回預測標籤

        return {
            "classification": "自殺" if prediction == 1 else "非自殺",
            "suicide_probability": probs[0][1].item(),
            "non_suicide_probability": probs[0][0].item()
        }

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
        result_text.config(text="請輸入一些文本。", fg="red")
        return

    result = ""
    if selected_model == "KNN":
        result = evaluator.predict_with_knn(text.split())
        result_text.config(text=f"分類: {result['classification']}\n自殺機率: {result['suicide_probability']*100:.2f}%\n非自殺機率: {result['non_suicide_probability']*100:.2f}%", fg="black")
    elif selected_model == "SVM (chiu)":
        result = evaluator.predict_with_svm1(text.split())
        result_text.config(text=f"分類: {result['classification']}\n自殺機率: {result['suicide_probability']*100:.2f}%\n非自殺機率: {result['non_suicide_probability']*100:.2f}%", fg="black")
    elif selected_model == "SVM (azz)":
        result = evaluator.predict_with_svm2(text)
        result_text.config(text=f"分類: {result['classification']}\n自殺機率: {result['suicide_probability']*100:.2f}%\n非自殺機率: {result['non_suicide_probability']*100:.2f}%", fg="black")
    elif selected_model == "邏輯回歸":
        result = evaluator.predict_with_logistic(text)
        result_text.config(text=f"分類: {result['classification']}\n自殺機率: {result['suicide_probability']*100:.2f}%\n非自殺機率: {result['non_suicide_probability']*100:.2f}%", fg="black")
    elif selected_model == "BERT":
        result = evaluator.predict_with_roberta(text)
        result_text.config(text=f"分類: {result['classification']}\n自殺機率: {result['suicide_probability']*100:.2f}%\n非自殺機率: {result['non_suicide_probability']*100:.2f}%", fg="black")
    else:
        result_text.config(text="請選擇一個模型", fg="red")
# 創建主窗口
root = tk.Tk()
root.title("自殺風險預測系統")
root.geometry('1200x800')
root.configure(bg='#f0f0f0')

# 設置全局字體
font_large = ("Arial", 16, "bold")
font_medium = ("Arial", 12)
font_small = ("Arial", 10)

# 標題標籤
title_label = tk.Label(root, text="自殺風險預測系統", font=font_large, bg='#4B9CD3', fg='white')
title_label.grid(row=0, column=0, columnspan=2, pady=20, sticky="ew")

# 輸入框標籤
label = tk.Label(root, text="輸入要分析的文本：", font=font_medium, bg='#f0f0f0')
label.grid(row=1, column=0, sticky="w", padx=20)

# 創建輸入框
entry = tk.Entry(root, font=font_medium, width=50, borderwidth=2, relief="solid")
entry.grid(row=2, column=0, padx=20, pady=10, ipadx=20, ipady=10, sticky="ew", columnspan=2)

# 創建模型選擇框
model_label = tk.Label(root, text="選擇預測模型：", font=font_medium, bg='#f0f0f0')
model_label.grid(row=3, column=0, sticky="w", padx=20)

model_var = tk.StringVar(value="KNN")
model_menu = ttk.Combobox(root, textvariable=model_var, font=font_small, state="readonly")
model_menu['values'] = ("KNN", "SVM (chiu)", "SVM (azz)", "邏輯回歸", "BERT")
model_menu.grid(row=4, column=0, padx=20, pady=10, sticky="ew", columnspan=2)

# 創建預測按鈕
predict_button = tk.Button(root, text="進行預測", font=font_medium, bg='#FF9800', fg='white', command=predict_risk)
predict_button.grid(row=5, column=0, padx=20, pady=20, ipadx=10, ipady=5, sticky="ew", columnspan=2)

# 結果顯示框架
result_frame = tk.Frame(root, bg='#FFFFFF', borderwidth=2, relief="solid")
result_frame.grid(row=6, column=0, padx=100, pady=100, sticky="ew", columnspan=5)

result_label = tk.Label(result_frame, text="結果：", font=("Arial", 18, "bold"), bg='#FFFFFF')
result_label.pack(pady=10)

result_text = tk.Label(result_frame, text="", font=("Arial", 20), bg='#FFFFFF', justify="left", wraplength=800)
result_text.pack(pady=5)

# 設置窗口可變大小調整
root.grid_columnconfigure(0, weight=1)

# 啟動GUI主循環
root.mainloop()

# %%
