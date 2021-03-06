import unittest

from resources.json_info.help import HelpMessage


class MyTestCase(unittest.TestCase):

    def test_kmeans_title_matches(self):
        help_message = HelpMessage("..\\resources\\json_info\\help_message.json")
        title, *_ = help_message["KMeans"]
        expected = "Estimador: Kmeans"
        self.assertEqual(title, expected)

    def test_no_feature_selection_url_matches(self):
        help_message = HelpMessage("..\\resources\\json_info\\help_message.json")
        *_, url = help_message["No_feature_selection"]
        expected = ""
        self.assertEqual(url, expected)

    def test_raise_key_error(self):
        with self.assertRaises(KeyError):
            help_message = HelpMessage("..\\resources\\json_info\\help_message.json")
            _ = help_message["invalid_key"]

    def test_raise_type_error(self):
        with self.assertRaises(TypeError):
            help_message = HelpMessage("..\\resources\\json_info\\help_message.json")
            _ = help_message[1]

    def test_path_is_wrong_raise_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            _ = HelpMessage("..\\resources\\json_info\\help.txt")

    def test_file_raises_type_error(self):
        with self.assertRaises(Exception):
            _ = HelpMessage(".\\..\\datasets\\iris.csv")


if __name__ == '__main__':
    unittest.main()
