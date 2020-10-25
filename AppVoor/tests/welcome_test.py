import unittest

from is_data import DataEnsurer
from jsonInfo.welcome import WelcomeMessenger, WelcomeMessage


class MyTestCase(unittest.TestCase):

    def test_messenger_message_is_str(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        # use of WelcomeMessenger with WelcomeMessage implementation
        welcome_messenger = WelcomeMessenger(welcome_message)
        message = str(welcome_messenger)
        # is the returned message a string?
        bol_answer = DataEnsurer.validate_py_data(message, str)
        # is the output a string?
        self.assertTrue(bol_answer)

    def test_test_messenger_message_is_str_message_len_is_greater_than_zero(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        # use of WelcomeMessenger with WelcomeMessage implementation
        welcome_messenger = WelcomeMessenger(welcome_message)
        message_len = len(welcome_messenger)  # gets message len using class method __len__
        bol_answer = DataEnsurer.validate_py_data(message_len, int)
        if bol_answer and message_len > 0:
            bol_answer = True  # it is indeed a integer and is greater than zero
        else:
            bol_answer = False  # it is indeed a integer, but the len is not greater than zero
        self.assertTrue(bol_answer)

    def test_welcome_message_author_matches(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        author, _ = welcome_message[0]  # gets the author by index using class method __getitem__
        expected_author = "Fei-Fei Li"
        # do the variables info match?
        self.assertEqual(author, expected_author)

    def test_welcome_message_quote_matches(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        _, quote = welcome_message[4]  # gets the quote by index using class method __getitem__
        expected_quote = "Lo que todos tenemos que hacer es asegurarnos de que estamos usando la IA de una manera que " \
                         "" \
                         "" \
                         "" \
                         "sea en beneficio de la humanidad, no en detrimento de la humanidad"
        # do the variables info match?
        self.assertEqual(quote, expected_quote)

    def test_data_from_json_is_tuple_positive_index(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        this_is_a_tuple = welcome_message[0]  # gets author and quote as tuple by index using class method __getitem__
        bol_answer = DataEnsurer.validate_py_data(this_is_a_tuple, tuple)
        # is the output a tuple?
        self.assertTrue(bol_answer)

    def test_data_from_json_is_tuple_negative_index(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        this_is_a_tuple = welcome_message[-5]  # gets author and quote as tuple by index using class method __getitem__
        bol_answer = DataEnsurer.validate_py_data(this_is_a_tuple, tuple)
        # is the output a tuple?
        self.assertTrue(bol_answer)

    def test_raise_index_error_when_get_data_by_index(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        with self.assertRaises(IndexError):
            _ = welcome_message[-6]  # __getitem__ method raises a error due to index out of bound


if __name__ == '__main__':
    unittest.main()
