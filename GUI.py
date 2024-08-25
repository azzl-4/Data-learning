# gui_app.py
#%%
import tkinter as tk
from tkinter import ttk, messagebox
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# 載入已保存的模型和 tokenizer
model = BertForSequenceClassification.from_pretrained('./suicide_detection_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 設置設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定義預測函數
def predict(text):
    encodings = tokenizer.encode_plus(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    encodings = {key: val.to(device) for key, val in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        prediction = torch.argmax(logits, dim=-1).cpu().item()
    return prediction, probs[0]

# 定義顯示預測結果的函數
def show_prediction():
    text = text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Error", "請輸入文本")
        return
    
    prediction, probs = predict(text)
    label = "自殺" if prediction == 1 else "非自殺"
    prob_str = [f"{prob*100:.2f}%" for prob in probs]
    
    result_label.config(text=f"預測標籤: {label}")
    prob_label.config(text=f"預測概率: 自殺: {prob_str[1]}, 非自殺: {prob_str[0]}")

# 設定 GUI
root = tk.Tk()
root.title("自殺風險判斷器")
root.configure(bg='#ffb6c1')  # 設置背景顏色

# 設定 GUI 內容
style = ttk.Style()
style.configure("TLabel", background='#f0f0f0', font=('Arial', 12))
style.configure("TButton", background='#4CAF50', foreground='black', font=('Arial', 12))
style.configure("TButton", padding=6)

# 設置 Grid 布局
root.grid_rowconfigure(1, weight=1)  # 文字框行自適應
root.grid_columnconfigure(0, weight=1)  # 文字框列自適應

ttk.Label(root, text="請輸入文本:").grid(row=0, column=0, padx=10, pady=10, sticky='w')

text_entry = tk.Text(root, width=50, height=10, bg='#000000', fg='#ffffff', font=('Arial', 12), borderwidth=2, relief="solid")
text_entry.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')  # nsew 讓文字框隨著視窗調整大小

predict_button = ttk.Button(root, text="判斷", command=show_prediction)
predict_button.grid(row=2, column=0, padx=10, pady=10, sticky='ew')  # 讓按鈕隨著視窗寬度調整

result_label = ttk.Label(root, text="預測標籤:")
result_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')

prob_label = ttk.Label(root, text="預測概率:")
prob_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')

# 啟動 GUI 主循環
root.mainloop()

# %%
