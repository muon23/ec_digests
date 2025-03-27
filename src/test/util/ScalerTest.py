import unittest

from util.Scaler import Scaler


class ScalerTest(unittest.TestCase):
    def test_to_unit(self):
        self.assertEqual(Scaler.to_unit("1.5", "thousand USD"), (1500.0, "USD"))
        self.assertEqual(Scaler.to_unit("2.3", "Million Records"), (2300000.0, "Records"))
        self.assertEqual(Scaler.to_unit("5", "k EUR"), (5000.0, "EUR"))
        self.assertEqual(Scaler.to_unit("1.2", "Billion"), (1200000000.0, ""))
        self.assertEqual(Scaler.to_unit("3", "K mL"), (3000.0, "mL"))
        self.assertEqual(Scaler.to_unit("2.3", "stuff"), (2.3, "stuff"))
        self.assertEqual(Scaler.to_unit("2", "trillion people"), (2_000_000_000_000.0, "people"))
        self.assertEqual(Scaler.to_unit("3", "something something"), (3.0, "something something"))

    def test_to_magnitude(self):
        self.assertEqual(Scaler.to_magnitude(4_000_000_000), (4, "billion"))
        self.assertEqual(Scaler.to_magnitude("4_000"), (4, 'thousand'))
        self.assertEqual(Scaler.to_magnitude(1_500_000), (1.5, 'million'))
        self.assertEqual(Scaler.to_magnitude("2_500_000_000_000"), (2.5, "trillion"))
        self.assertEqual(Scaler.to_magnitude("-2_500_000_000_000"), (-2.5, "trillion"))
        self.assertEqual(Scaler.to_magnitude("-13"), (-13, ""))

    def test_normalize_currency(self):
        self.assertEqual(Scaler.normalize_currency(4_000_000_000, "USD"), ("4", "billion USD"))
        self.assertEqual(Scaler.normalize_currency(4_000, "EUR million"), ("4", "billion EUR"))
        self.assertEqual(Scaler.normalize_currency(1_500_000, "GBP"), (None, None))  # GBP not supported
        self.assertEqual(Scaler.normalize_currency("1_500_000", "GBP", more_codes="GBP"), ("1.5", "million GBP"))
        self.assertEqual(Scaler.normalize_currency("abcd", "USD"), (None, None))
        self.assertEqual(Scaler.normalize_currency("10", None), (None, None))

    def test_normalize_count(self):
        self.assertEqual(Scaler.normalize_count(4_000_000_000, "people"), ("4", "billion people"))
        self.assertEqual(Scaler.normalize_count(4_100_000_000, "ea"), ("4.1", "billion"))
        self.assertEqual(Scaler.normalize_count("3.500", None), ("3.5", ""))

    def test_normalize_multiple(self):
        self.assertEqual(Scaler.normalize_multiple("4x", None), ("4", "times"))
        self.assertEqual(Scaler.normalize_multiple("4", "X"), ("4", "times"))
        self.assertEqual(Scaler.normalize_multiple("4100", "X"), ("4.1", "thousand times"))
        self.assertEqual(Scaler.normalize_currency("10", None), (None, None))

    def test_normalize(self):
        self.assertEqual(Scaler.normalize(4_000_000_000, "USD"), ("4", "billion USD"))
        self.assertEqual(Scaler.normalize(4_100_000_000, "ea"), ("4.1", "billion"))
        self.assertEqual(Scaler.normalize("4100", "X"), ("4.1", "thousand times"))
        self.assertEqual(Scaler.normalize("up", "up"), ("up", "up"))

if __name__ == '__main__':
    unittest.main()
