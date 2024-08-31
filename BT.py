#%%
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm
import numpy as np

# 自定义数据集类
class SuicideDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_roberta():
    # 加载数据
    data = pd.read_csv('Suicide_Detection_expanded.csv')
    data = data.drop(columns=['Unnamed: 0'])
    data['class'] = data['class'].map({'suicide': 1, 'non-suicide': 0})

    # 分割数据为训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'].tolist(),
        data['class'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=data['class']
    )

    # 初始化 RoBERTa 分词器
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # 创建训练和验证数据集
    train_dataset = SuicideDataset(train_texts, train_labels, tokenizer, max_len=128)
    val_dataset = SuicideDataset(val_texts, val_labels, tokenizer, max_len=128)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    # 初始化 RoBERTa 模型
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model = model.cuda()

    # 定义优化器、学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 训练循环
    num_epochs = 3
    train_losses = []
    val_losses = []
    all_train_preds = []
    all_train_labels = []
    all_val_preds = []
    all_val_labels = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

            preds = torch.argmax(outputs.logits, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_losses.append(total_train_loss / len(train_loader))

        # 验证集评估
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['label'].cuda()

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_losses.append(total_val_loss / len(val_loader))
        print(f'Epoch {epoch} - Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # 保存训练好的模型
    model.save_pretrained('./suicide_detection_roberta_model')
    tokenizer.save_pretrained('./suicide_detection_roberta_model')

    # 保存预测结果
    np.savez('roberta_results.npz', train_labels=all_train_labels, train_preds=all_train_preds, 
                                      val_labels=all_val_labels, val_preds=all_val_preds,
                                      train_losses=train_losses, val_losses=val_losses)

    # 完整验证集评估
    print(classification_report(all_val_labels, all_val_preds, target_names=['Non-Suicide', 'Suicide']))

    # 使用训练数据生成 BERT 特征并拟合 PCA
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Generating BERT features"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            outputs = model.roberta(input_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_features.append(features)

    # 将特征数组合并
    all_features = np.vstack(all_features)

    # 拟合 PCA 并保存
    pca = PCA(n_components=100)
    pca.fit(all_features)
    joblib.dump(pca, 'pca_model.pkl')
    print("PCA model saved as 'pca_model.pkl'")

if __name__ == "__main__":
    train_roberta()

# %%
