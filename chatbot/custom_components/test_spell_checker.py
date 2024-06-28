import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spell_checker import SpellCheckerComponent

def test_correct_spelling():
    component = SpellCheckerComponent({})
    corrected_text = component.correct_spelling("hte recieve")
    assert corrected_text == "the receive"
    print("Test passed!")

test_correct_spelling()
