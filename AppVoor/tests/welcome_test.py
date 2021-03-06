import unittest

from resources.backend_scripts.is_data import DataEnsurer
from resources.json_info.welcome import WelcomeMessenger, WelcomeMessage


class MyTestCase(unittest.TestCase):

    def test_messenger_message_is_str(self):
        # use of WelcomeMessenger with WelcomeMessage implementation
        welcome_messenger = WelcomeMessenger(file_path="..\\resources\\json_info\\welcome_message.json")
        message = str(welcome_messenger)
        # is the returned message a string?
        bol_answer = DataEnsurer.validate_py_data(message, str)
        # is the output a string?
        self.assertTrue(bol_answer)

    def test_test_messenger_message_is_str_message_len_is_greater_than_zero(self):
        # use of WelcomeMessenger with WelcomeMessage implementation
        welcome_messenger = WelcomeMessenger(file_path="..\\resources\\json_info\\welcome_message.json")
        message_len = len(welcome_messenger)  # gets message len using class method __len__
        bol_answer = DataEnsurer.validate_py_data(message_len, int)
        if bol_answer and message_len > 0:
            bol_answer = True  # it is indeed a integer and is greater than zero
        else:
            bol_answer = False  # it is indeed a integer, but the len is not greater than zero
        self.assertTrue(bol_answer)

    def test_welcome_message_author_matches(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path="..\\resources\\json_info\\welcome_message.json", data_type=list)
        author, _ = welcome_message[0]  # gets the author by index using class method __getitem__
        expected_author = "Fei-Fei Li"
        # do the variables info match?
        self.assertEqual(author, expected_author)

    def test_welcome_message_quote_matches(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path="..\\resources\\json_info\\welcome_message.json", data_type=list)
        _, quote = welcome_message[4]  # gets the quote by index using class method __getitem__
        expected_quote = "Lo que todos tenemos que hacer es asegurarnos de que estamos usando la IA de una manera que "\
                         "sea en beneficio de la humanidad, no en detrimento de la humanidad"
        # do the variables info match?
        self.assertEqual(quote, expected_quote)

    def test_data_from_json_is_tuple_positive_index(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path="..\\resources\\json_info\\welcome_message.json", data_type=list)
        this_is_a_tuple = welcome_message[0]  # gets author and quote as tuple by index using class method __getitem__
        bol_answer = DataEnsurer.validate_py_data(this_is_a_tuple, tuple)
        # is the output a tuple?
        self.assertTrue(bol_answer)

    def test_get_item_index_is_not_int(self):
        with self.assertRaises(TypeError):
            # initialization of welcome_message with its path
            welcome_message = WelcomeMessage(file_path="..\\resources\\json_info\\welcome_message.json", data_type=list)
            _ = welcome_message['s']  # key is not a valid type. It should be an integer

    def test_welcome_message_wrong_type(self):
        with self.assertRaises(TypeError):
            # initialization of welcome_message with its path. data_type is wrong
            _ = WelcomeMessage(file_path="..\\resources\\json_info\\welcome_message.json", data_type=str)

    def test_data_from_json_is_tuple_negative_index(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path="..\\resources\\json_info\\welcome_message.json", data_type=list)
        this_is_a_tuple = welcome_message[-5]  # gets author and quote as tuple by index using class method __getitem__
        bol_answer = DataEnsurer.validate_py_data(this_is_a_tuple, tuple)
        # is the output a tuple?
        self.assertTrue(bol_answer)

    def test_raise_index_error_when_get_data_by_index(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path="..\\resources\\json_info\\welcome_message.json", data_type=list)
        with self.assertRaises(IndexError):
            _ = welcome_message[-6]  # __getitem__ method raises a error due to index out of bounds


if __name__ == '__main__':
    unittest.main()
