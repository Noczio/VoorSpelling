import unittest

from is_data import is_str, is_int, is_tuple
from jsonInfo.welcome import WelcomeMessenger, WelcomeMessage


class MyTestCase(unittest.TestCase):

    def test_messenger_message_is_str(self):
        welcome_messenger = WelcomeMessenger(message_path=".\\..\\jsonInfo\\welcomeMessage.json")
        message = str(welcome_messenger)
        bol_answer = is_str(message)
        self.assertTrue(bol_answer)

    def test_test_messenger_message_is_str_message_len_is_greater_than_zero(self):
        welcome_messenger = WelcomeMessenger(message_path=".\\..\\jsonInfo\\welcomeMessage.json")
        message_len = len(welcome_messenger)
        bol_answer = is_int(message_len)
        if bol_answer and message_len > 0:
            bol_answer = True
        else:
            bol_answer = False
        self.assertTrue(bol_answer)

    def test_welcome_message_author_matches(self):
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        author, _ = welcome_message.get_data_by_index(0)
        expected_author = "Fei-Fei Li"
        self.assertEqual(author, expected_author)

    def test_welcome_message_quote_matches(self):
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        _, quote = welcome_message.get_data_by_index(4)
        expected_quote = "Lo que todos tenemos que hacer es asegurarnos de que estamos usando la IA de una manera que " \
                         "sea en beneficio de la humanidad, no en detrimento de la humanidad"
        self.assertEqual(quote, expected_quote)

    def test_data_from_json_is_tuple_positive_index(self):
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        this_is_a_tuple = welcome_message.get_data_by_index(0)
        bol_answer = is_tuple(this_is_a_tuple)
        self.assertTrue(bol_answer)

    def test_data_from_json_is_tuple_negative_index(self):
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        this_is_a_tuple = welcome_message.get_data_by_index(-5)
        bol_answer = is_tuple(this_is_a_tuple)
        self.assertTrue(bol_answer)

    def test_raise_index_error_when_get_data_by_index(self):
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        with self.assertRaises(IndexError):
            _ = welcome_message.get_data_by_index(-6)


if __name__ == '__main__':
    unittest.main()
