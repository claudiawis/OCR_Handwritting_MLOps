import unittest
import os
import pandas as pd
from src.data.load_dataset import load_images

class TestLoadDataset(unittest.TestCase):

    def setUp(self):
        # Create temporary test directories and files
        self.test_base_path = "tests/test_data/tmp_words"
        self.test_words_file = "tests/test_data/tmp_words.txt"
        os.makedirs(self.test_base_path, exist_ok=True)

        # Create a dummy words.txt file
        with open(self.test_words_file, "w") as f:
            f.write("# Comment line\n")
            f.write("a01-000u-00-00 ok 128 1779 444 768 1772 1993 The\n")
            f.write("a01-000u-01-00 ok 128 1784 442 751 1778 1986 quick\n")

        # Create dummy image files
        os.makedirs(os.path.join(self.test_base_path, "a01"), exist_ok=True)
        os.makedirs(os.path.join(self.test_base_path, "a01/a01-000u"), exist_ok=True)
        with open(os.path.join(self.test_base_path, "a01/a01-000u/a01-000u-00-00.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # Minimal PNG header
        with open(os.path.join(self.test_base_path, "a01/a01-000u/a01-000u-01-00.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # Minimal PNG header

    def tearDown(self):
        # Clean up temporary files and directories
        if os.path.exists(self.test_base_path):
            for root, dirs, files in os.walk(self.test_base_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.test_base_path)

        if os.path.exists(self.test_words_file):
            os.remove(self.test_words_file)

    def test_load_images(self):
        # Call the function and test its output
        df = load_images(self.test_base_path, self.test_words_file)

        # Check the resulting DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)  # Ensure both rows are included
        self.assertListEqual(
            list(df.columns),
            ['line_id', 'result', 'graylevel', 'x', 'y', 'w', 'h', 'annotation', 'transcription', 'image_path']
        )
        self.assertIn("quick", df["transcription"].values)  # Verify transcription is correct

if __name__ == "__main__":
    unittest.main()
