import unittest
from unittest.mock import MagicMock, call, patch

import tools


class InternetSearchRetryTests(unittest.TestCase):
    @patch.object(tools.time, "sleep")
    @patch.object(tools, "TavilyClient")
    def test_search_waits_two_then_four_seconds(self, client_class, sleep_mock):
        client = MagicMock()
        client.search.side_effect = [
            RuntimeError("first failure"),
            RuntimeError("second failure"),
            {"results": [{"title": "success"}]},
        ]
        client_class.return_value = client

        results = tools.internet_search.func("test query")

        self.assertEqual(
            results,
            [
                {
                    "title": "success",
                    "url": "",
                    "content": "",
                    "published_date": None,
                    "score": None,
                }
            ],
        )
        self.assertEqual(client.search.call_count, 3)
        self.assertTrue(
            all(
                search_call.kwargs["max_results"] == 3
                for search_call in client.search.call_args_list
            )
        )
        self.assertEqual(sleep_mock.call_args_list, [call(2), call(4)])

    @patch.object(tools, "TavilyClient")
    def test_search_compacts_provider_content(self, client_class):
        client = MagicMock()
        client.search.return_value = {
            "results": [
                {
                    "title": "Evidence",
                    "url": "https://example.com",
                    "content": "x" * 2_000,
                    "score": 0.9,
                }
            ]
        }
        client_class.return_value = client

        results = tools.internet_search.func("test query", max_results=20)

        self.assertEqual(client.search.call_args.kwargs["max_results"], 3)
        self.assertEqual(len(results[0]["content"]), 600)


if __name__ == "__main__":
    unittest.main()
