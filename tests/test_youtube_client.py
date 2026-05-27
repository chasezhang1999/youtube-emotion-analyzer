import unittest

from youtube_emotion.youtube_client import fetch_top_comments


class FakeResponse:
    status_code = 200
    text = "ok"

    def json(self):
        return {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {
                                "textOriginal": "  Great&nbsp;video\n",
                            }
                        }
                    }
                },
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {
                                "textDisplay": "Interesting campaign",
                            }
                        }
                    }
                },
            ]
        }


class YouTubeClientTest(unittest.TestCase):
    def test_fetch_top_comments_uses_expected_api_parameters(self):
        calls = []

        def fake_get(url, params, timeout):
            calls.append({"url": url, "params": params, "timeout": timeout})
            return FakeResponse()

        comments = fetch_top_comments(
            video_id="dQw4w9WgXcQ",
            api_key="test-key",
            max_results=100,
            order="relevance",
            request_get=fake_get,
        )

        self.assertEqual(comments, ["Great video", "Interesting campaign"])
        self.assertEqual(calls[0]["params"]["part"], "snippet")
        self.assertEqual(calls[0]["params"]["videoId"], "dQw4w9WgXcQ")
        self.assertEqual(calls[0]["params"]["maxResults"], 100)
        self.assertEqual(calls[0]["params"]["order"], "relevance")
        self.assertEqual(calls[0]["params"]["textFormat"], "plainText")

    def test_fetch_top_comments_requires_api_key(self):
        with self.assertRaises(ValueError):
            fetch_top_comments("dQw4w9WgXcQ", "")


if __name__ == "__main__":
    unittest.main()
