import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification

model_path = './saved_model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

def preprocess_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def predict_emotion(text):
    preprocessed_text = preprocess_text(text)
    inputs = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()


if __name__ == "__main__":
    input_text = input("Unesite tekst: ")
    emotion = predict_emotion(input_text)
    print(f"Predikovana emocija: {emotion}")
