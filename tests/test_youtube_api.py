import sys
import os
import unittest

# Add the parent directory to the system path so the youtube_api module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from youtube_api import fetch_top_videos  # Import the function from youtube_api.py


class TestYouTubeAPI(unittest.TestCase):
    def test_fetch_top_videos(self):
        # Replace with a valid YouTube API key
        videos = fetch_top_videos(
            "AIzaSyA3DTXw6hALW-JnDmjFmJcXCIvJE5mFMNQ", "Python tutorial"
        )
        self.assertTrue(len(videos) > 0)  # Ensure there are results


if __name__ == "__main__":
    unittest.main()
