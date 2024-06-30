from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from custom_components.emotion_analyzer import EmotionAnalyzer

class ActionGoodbye(Action):

    def name(self) -> Text:
        return "action_goodbye"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="Goodbye! Have a great day!")
        return []

class ActionBotChallenge(Action):

    def name(self) -> Text:
        return "action_bot_challenge"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="I'm a bot, powered by Rasa.")
        return []

class ActionRespondBasedOnEmotion(Action):
    def name(self) -> Text:
        return "action_respond_based_on_emotion"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        intent = tracker.latest_message['intent'].get('name')
        emotion = tracker.latest_message.get('emotion', 'neutral')
        
        print(f"Detected intent: {intent}")
        print(f"Detected emotion: {emotion}")
        
        if emotion == 'positive':
            action_name = f"utter_{intent}_positive"
        else:
            action_name = f"utter_{intent}_negative"
        
        print(f"Selected action: {action_name}")
        
        dispatcher.utter_message(response=action_name)
        
        return []