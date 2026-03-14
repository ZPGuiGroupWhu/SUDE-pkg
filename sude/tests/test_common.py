import unittest

from sude import __version__, sude


class TestPublicAPI(unittest.TestCase):
    def test_public_exports(self):
        self.assertTrue(callable(sude))
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
