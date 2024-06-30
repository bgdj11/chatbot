import re
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Dict, Text, Any, List
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.graph import ExecutionContext

@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False)
class EmotionAnalyzer(GraphComponent):
    def __init__(self, config: Dict[Text, Any]) -> None:
        try:
            print("Initializing EmotionAnalyzer...")
            model_path = config.get("model_path", os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'saved_model'))
            print(f"Model path: {model_path}")
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise e

    def preprocess_text(self, text):
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def predict_emotion(self, text):
        preprocessed_text = self.preprocess_text(text)
        inputs = self.tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.item()

    def process(self, messages: List[Message], **kwargs: Any) -> List[Message]:
        for message in messages:
            text = message.get("text")
            emotion = self.predict_emotion(text)
            message.set("emotion", emotion, add_to_output=True)
        return messages

    def train(self, training_data: TrainingData, **kwargs: Any) -> None:
        pass

    def process_training_data(self, training_data: TrainingData, **kwargs: Any) -> TrainingData:
        return training_data

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        print("Creating EmotionAnalyzer component with config:", config)
        return cls(config)
