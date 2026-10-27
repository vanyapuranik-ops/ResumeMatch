import unittest

from src.predict import assess_input, confidence_notes, LOW_CONFIDENCE


class TestResumeInputValidation(unittest.TestCase):

    def test_empty_input_is_refused(self):
        result = assess_input("")
        self.assertIsNotNone(result["refuse"])
        self.assertEqual(result["words"], 0)

    def test_whitespace_input_is_refused(self):
        result = assess_input("   \n\t ")
        self.assertIsNotNone(result["refuse"])
        self.assertEqual(result["words"], 0)

    def test_49_words_are_refused(self):
        result = assess_input(" ".join(["experience"] * 49))
        self.assertIsNotNone(result["refuse"])
        self.assertEqual(result["words"], 49)

    def test_50_words_are_allowed_with_warning(self):
        result = assess_input(" ".join(["experience"] * 50))
        self.assertIsNone(result["refuse"])
        self.assertTrue(result["warnings"])

    def test_112_words_warn_about_training_minimum(self):
        result = assess_input(" ".join(["experience"] * 112))
        self.assertIsNone(result["refuse"])
        self.assertTrue(result["warnings"])
        self.assertIn("shortest had 113", result["warnings"][0])

    def test_113_words_warn_about_short_resume(self):
        result = assess_input(" ".join(["experience"] * 113))
        self.assertIsNone(result["refuse"])
        self.assertTrue(result["warnings"])

    def test_249_words_warn(self):
        result = assess_input(" ".join(["experience"] * 249))
        self.assertTrue(result["warnings"])

    def test_250_words_no_length_warning(self):
        result = assess_input(" ".join(["experience"] * 250))
        self.assertIsNone(result["refuse"])
        self.assertFalse(result["warnings"])

    def test_numeric_punctuation_heavy_text_warns(self):
        result = assess_input(" ".join(["12345 !!!"] * 60))
        self.assertIsNone(result["refuse"])
        self.assertTrue(any("characters are letters" in w
                            for w in result["warnings"]))

    def test_low_confidence_below_threshold(self):
        notes = confidence_notes([{"category": "ARTS", "score": 0.6999}])
        self.assertTrue(any("LOW CONFIDENCE" in n for n in notes))

    def test_low_confidence_at_threshold(self):
        notes = confidence_notes([{"category": "ARTS", "score": 0.70}])
        self.assertTrue(any("LOW CONFIDENCE" in n for n in notes))

    def test_above_threshold_has_no_low_confidence_warning(self):
        notes = confidence_notes([{"category": "ARTS", "score": 0.7001}])
        self.assertFalse(any("LOW CONFIDENCE" in n for n in notes))
        self.assertTrue(any("not calibrated probabilities" in n for n in notes))

    def test_threshold_is_provisional(self):
        notes = confidence_notes([{"category": "ARTS", "score": 0.50}])
        self.assertTrue(any("provisional" in n for n in notes))

    def test_scores_are_described_as_uncalibrated(self):
        notes = confidence_notes([{"category": "ARTS", "score": 0.90}])
        self.assertTrue(any("not calibrated probabilities" in n for n in notes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
