#%%
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# 加载已保存的模型和 tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./saved_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')

# 加载标签编码器
label_encoder = joblib.load('./saved_model/label_encoder.joblib')

# 将模型移动到 GPU (如有)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 示例测试文本
test_texts = [
    "I feel overwhelmed by pain and want to end everything.",
    "I've been thinking about how to commit suicide lately; I feel so hopeless.",
    "I've been feeling really down lately, and things are tough.",
    "Today is a beautiful day and I feel happy."
]

# 编码测试文本
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = {key: val.to(device) for key, val in test_encodings.items()}

# 模型推理
with torch.no_grad():
    test_outputs = model(**test_encodings)
    test_logits = test_outputs.logits

# 获取预测结果
test_predictions = torch.argmax(test_logits, dim=-1)
test_predicted_labels = test_predictions.cpu().numpy()

# 如果需要将预测标签转换回原始文字标签
predicted_labels_text = label_encoder.inverse_transform(test_predicted_labels)
print(f"Predicted Labels (text): {predicted_labels_text}")

for text, label in zip(test_texts, predicted_labels_text):
    print(f"Text: {text}\nPredicted Label: {label}\n")

# %%
