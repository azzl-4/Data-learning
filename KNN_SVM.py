import pandas as pd
from gensim.utils import simple_preprocess
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取數據並隨機選取50000筆
file_path = 'Suicide_Detection.csv'
df = pd.read_csv(file_path)
#df = dff.sample(n=50000, random_state=42)
print(df.head())
print(df['class'].value_counts())

sentences = df['text']
labels = df['class']

# 分词
tokenized_sentences = sentences.apply(simple_preprocess)

# 加载 Word2Vec 模型和 TF-IDF 字典
model = joblib.load('word2vec_model.pkl')
tfidf_dict = joblib.load('tfidf_dict.pkl')

# 函数：将句子转化为TF-IDF加权平均的向量
def sentence_to_vector(sentence, model, tfidf_dict):
    vectors = []
    weights = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])
            weights.append(tfidf_dict.get(word, 1.0))
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.average(vectors, axis=0, weights=weights)

# 将 Series 中的每个句子转换为向量
sentence_vectors = np.array([sentence_to_vector(sentence, model, tfidf_dict) for sentence in tokenized_sentences])

# 标签编码
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(sentence_vectors, encoded_labels, test_size=0.3, random_state=42)

# 加载已保存的 KNN 和 SVM 模型
best_knn = joblib.load('best_knn_model.pkl')
best_svm = joblib.load('best_svm_model.pkl')

# 混淆矩阵和分类报告
def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} - Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.show()
    
    print(f"\n{model_name} - Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 评估 KNN 和 SVM 模型
evaluate_model(best_knn, X_test, y_test, label_encoder, "KNN")
evaluate_model(best_svm, X_test, y_test, label_encoder, "SVM")