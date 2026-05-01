# tests/test_virtual_subject.py

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from svomptr.layers.virtual_subject import VirtualSubjectDetector, VirtualSubjectInfo

class TestVirtualSubject(unittest.TestCase):
    
    def setUp(self):
        self.detector = VirtualSubjectDetector()
    
    def test_existential_there_is(self):
        """Test 'There is' pattern detection"""
        is_virtual, info = self.detector.detect(["there", "is", "a", "cat"])
        self.assertTrue(is_virtual)
        self.assertEqual(info.type, "existential")
        self.assertEqual(info.word, "there")
        self.assertEqual(info.introduced_noun, "cat")
    
    def test_existential_there_are(self):
        """Test 'There are' pattern"""
        is_virtual, info = self.detector.detect(["there", "are", "many", "people"])
        self.assertTrue(is_virtual)
        self.assertEqual(info.type, "existential")
    
    def test_weather_it_is_raining(self):
        """Test 'It is raining' as weather (no subject)"""
        is_virtual, info = self.detector.detect(["it", "is", "raining"])
        self.assertTrue(is_virtual)
        self.assertEqual(info.type, "dummy_weather")
        self.assertEqual(info.verb, "raining")
    
    def test_reference_it_is_red(self):
        """Test 'It is red' referring to previous subject"""
        # Set context first
        self.detector.last_introduced_noun = "car"
        is_virtual, info = self.detector.detect(["it", "is", "red"])
        self.assertTrue(is_virtual)
        self.assertEqual(info.type, "dummy_general")
        self.assertEqual(info.refers_to, "car")
    
    def test_myanmar_weather(self):
        """Test Myanmar weather pattern"""
        is_virtual, info = self.detector.detect(["မိုး", "ရွာနေ", "တယ်"])
        self.assertTrue(is_virtual)
        self.assertEqual(info.type, "dummy_weather")
    
    def test_pronoun_resolution(self):
        """Test 'it' resolution works correctly"""
        self.detector.last_introduced_noun = "book"
        resolved = self.detector.resolve_it(
            VirtualSubjectInfo(is_virtual=True, type="dummy_general", word="it")
        )
        self.assertEqual(resolved, "book")

if __name__ == "__main__":
    unittest.main()
