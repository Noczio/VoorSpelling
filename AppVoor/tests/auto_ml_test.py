import unittest

from auto_ml import JarAutoML


class MyTestCase(unittest.TestCase):

    def test_jar_creation_n_folds_raises_error(self):
        # n_folds must be between 3 and 10
        with self.assertRaises(ValueError):
            _ = JarAutoML(-1, False, 10)

    def test_jar_creation_max_rand_raises_error(self):
        # max_rand must be grater or equal to 0
        with self.assertRaises(ValueError):
            _ = JarAutoML(3, False, -1)

    def test_random_state_is_right_when_max_rand_is_zero(self):
        # create a JarAutoML objet with max_rand = 0
        auto_ml = JarAutoML(3, False, 0)
        rand_state = auto_ml._random_state
        self.assertEqual(rand_state, 0)

    def test_random_state_is_right_when_max_rand_is_4000(self):
        # create a JarAutoML objet with max_rand = 4000
        max_rand = 4000
        auto_ml = JarAutoML(3, False, max_rand)
        rand_state = auto_ml._random_state
        # check if random state is the valid range
        if 0 <= rand_state <= 4000:
            bol_answer = True
        else:
            bol_answer = False
        # this should be True
        self.assertTrue(bol_answer)


if __name__ == '__main__':
    unittest.main()
