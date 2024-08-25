#%%
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SuicideRiskEvaluator:
    def __init__(self, model_path, tokenizer_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def evaluate_sentence(self, sentence, neutral_threshold=0.1):
        # Tokenize the input sentence
        encoding = self.tokenizer.encode_plus(
            sentence,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        prob = probabilities.cpu().numpy()[0]

        # Determine the class
        if abs(prob[0] - prob[1]) < neutral_threshold:
            classification = "Neutral"
        else:
            classification = "Suicide" if prob[1] > prob[0] else "Non-suicide"

        return {
            'text': sentence,
            'classification': classification,
            'suicide_probability': prob[1],
            'non_suicide_probability': prob[0]
        }

    def evaluate_sentences(self, sentences, neutral_threshold=0.1):
        results = []
        for sentence in sentences:
            result = self.evaluate_sentence(sentence, neutral_threshold)
            results.append(result)
        return results

if __name__ == "__main__":
    # Initialize evaluator with the trained model path
    model_path = './suicide_detection_model'
    evaluator = SuicideRiskEvaluator(model_path)

    # Example sentences to evaluate
    sentences = [
        "I feel like I can't go on any longer. Everything seems hopeless.",
        "I don't see a way out of this pain. I'm contemplating ending my life.",
        "I'm overwhelmed with sadness and can't see a future for myself.",
        "Today was a good day. I had a great time with friends.",
        "I am excited about my upcoming vacation. Things are looking positive.",
        "I had a productive day at work and feel good about my progress.",
        "I sometimes think about giving up, but I want to keep fighting."
    ]

    # Evaluate the sentences
    results = evaluator.evaluate_sentences(sentences)

    # Print the results
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Classification: {result['classification']}")
        print(f"Suicide Probability: {result['suicide_probability']*100:.2f}%")
        print(f"Non-suicide Probability: {result['non_suicide_probability']*100:.2f}%")
        print('---')

# %%
