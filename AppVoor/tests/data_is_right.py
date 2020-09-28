import unittest

from parse_file import DataParseEnsurer


class MyTestCase(unittest.TestCase):
    def test_data_is_int(self):
        ensurer_bol = DataParseEnsurer.is_int(10)
        self.assertTrue(ensurer_bol)

    def test_data_is_str(self):
        ensurer_bol = DataParseEnsurer.is_str("test")
        self.assertTrue(ensurer_bol)

    def test_data_is_float(self):
        ensurer_bol = DataParseEnsurer.is_float(0.25)
        self.assertTrue(ensurer_bol)

    def test_data_is_list(self):
        ensurer_bol = DataParseEnsurer.is_list([10, "s", True])
        self.assertTrue(ensurer_bol)

    def test_data_is_tuple(self):
        ensurer_bol = DataParseEnsurer.is_tuple((10, "s", True))
        self.assertTrue(ensurer_bol)


if __name__ == '__main__':
    unittest.main()
