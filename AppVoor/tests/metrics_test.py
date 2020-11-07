import unittest

from jsonInfo.metrics import CVMetrics


class MyTestCase(unittest.TestCase):
    def test_metrics_match(self):
        expected = [
            "accuracy",
            "f1",
            "roc_auc",
            "mutual_info_score",
            "r2",
            "precision"
        ]
        # create a CVMetrics with its path
        cv_metrics = CVMetrics(file_path=".\\..\\jsonInfo\\CVMetrics.json", data_type=list)
        # get the data
        metrics = cv_metrics.data
        # get in a list if every value is a True or False
        results = [e == m for e, m in zip(expected, metrics)]
        # are all outputs True?
        for r in results:
            self.assertTrue(r)

    def test_metric_key_zero_is_right(self):
        # create a CVMetrics with its path
        cv_metrics = CVMetrics(file_path=".\\..\\jsonInfo\\CVMetrics.json", data_type=list)
        # get data by key
        metric = cv_metrics[0]
        expected = "accuracy"
        # those values should be the same
        self.assertEqual(metric, expected)

    def test_metric_key_zero_raises_IndexError(self):
        # create a CVMetrics with its path
        cv_metrics = CVMetrics(file_path=".\\..\\jsonInfo\\CVMetrics.json", data_type=list)
        with self.assertRaises(IndexError):
            # get data by key
            _ = cv_metrics[-20]

    def test_metric_key_raises_TypeError(self):
        # create a CVMetrics with its path
        cv_metrics = CVMetrics(file_path=".\\..\\jsonInfo\\CVMetrics.json", data_type=list)
        with self.assertRaises(TypeError):
            # get data by key
            _ = cv_metrics[1.0]


if __name__ == '__main__':
    unittest.main()
