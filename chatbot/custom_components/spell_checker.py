from typing import Any, Dict, List, Text
from rasa.engine.graph import GraphComponent, GraphNode
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import os

@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False)
class SpellCheckerComponent(GraphComponent):
    def __init__(self, config: Dict[Text, Any], **kwargs: Any) -> None:
        super().__init__()
        self.config = config
        corrections_filepath = os.path.join(os.path.dirname(__file__), 'corrections.txt')
        self.corrections = self.load_corrections(corrections_filepath)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: Any,
        resource: Any,
        execution_context: Any,
    ) -> GraphNode:
        return cls(config)

    def load_corrections(self, filepath: str) -> Dict[str, str]:
        corrections = {}
        if os.path.exists(filepath):
            print(f"Loading corrections from: {filepath}")
            with open(filepath, "r") as file:
                for line in file:
                    line = line.strip()
                    if "," in line:
                        incorrect, correct = line.split(",", 1)
                        corrections[incorrect.strip()] = correct.strip()
                    else:
                        print(f"Line '{line}' is not properly formatted.")
        else:
            print(f"File '{filepath}' does not exist.")
        print(f"Loaded corrections: {corrections}")
        return corrections

    def correct_spelling(self, text: str) -> str:
        if not text:
            return text
        words = text.split()
        corrected_words = [self.corrections.get(word, word) for word in words]
        return " ".join(corrected_words)

    def process(self, messages: List[Message], **kwargs: Any) -> List[Message]:
        for message in messages:
            text = message.get("text")
            print(f"Original text: {text}")
            if text:
                corrected_text = self.correct_spelling(text)
                message.set("text", corrected_text)
                print(f"Corrected text: {corrected_text}")
        return messages

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        for example in training_data.training_examples:
            text = example.get("text")
            if text:
                corrected_text = self.correct_spelling(text)
                example.set("text", corrected_text)
        return training_data
