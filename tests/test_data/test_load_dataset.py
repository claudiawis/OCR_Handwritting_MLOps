import unittest
import os
import pandas as pd
import logging
from src.data.load_dataset import load_images

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLoadDataset(unittest.TestCase):

    def setUp(self):
        # Create temporary test directories and files
        self.test_base_path = "tests/test_data/tmp_words"
        self.test_words_file = "tests/test_data/tmp_words.txt"
        os.makedirs(self.test_base_path, exist_ok=True)

        logger.info("Creating temporary base path: %s", self.test_base_path)

        # Create a dummy words.txt file
        with open(self.test_words_file, "w") as f:
            f.write("# Comment line\n")
            f.write("a01-000u-00-00 ok 128 1779 444 768 1772 1993 The\n")
            f.write("a01-000u-01-00 ok 128 1784 442 751 1778 1986 quick\n")
        logger.info("Temporary words file created: %s", self.test_words_file)

        # Create dummy image files
        os.makedirs(os.path.join(self.test_base_path, "a01"), exist_ok=True)
        os.makedirs(os.path.join(self.test_base_path, "a01/a01-000u"), exist_ok=True)
        with open(os.path.join(self.test_base_path, "a01/a01-000u/a01-000u-00-00.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # Minimal PNG header
        logger.info("Dummy image file created: %s/a01/a01-000u/a01-000u-00-00.png", self.test_base_path)
        with open(os.path.join(self.test_base_path, "a01/a01-000u/a01-000u-01-00.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # Minimal PNG header
        logger.info("Dummy image file created: %s/a01/a01-000u/a01-000u-01-00.png", self.test_base_path)

    def tearDown(self):
        # Clean up temporary files and directories
        if os.path.exists(self.test_base_path):
            logger.info("Cleaning up temporary base path: %s", self.test_base_path)
            for root, dirs, files in os.walk(self.test_base_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                    logger.info("Removed file: %s", os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.test_base_path)
            logger.info("Removed base path: %s", self.test_base_path)

        if os.path.exists(self.test_words_file):
            os.remove(self.test_words_file)
            logger.info("Removed temporary words file: %s", self.test_words_file)

    def test_load_images(self):
        # Call the function and test its output
        logger.info("Loading images from: %s with words file: %s", self.test_base_path, self.test_words_file)
        df = load_images(self.test_base_path, self.test_words_file)

        # Check the resulting DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        logger.info("Loaded data is a DataFrame.")

        self.assertEqual(len(df), 2)  # Ensure both rows are included
        logger.info("DataFrame contains %d rows as expected.", len(df))

        self.assertListEqual(
            list(df.columns),
            ['line_id', 'result', 'graylevel', 'x', 'y', 'w', 'h', 'annotation', 'transcription', 'image_path']
        )
        logger.info("DataFrame columns match expected values.")

        self.assertIn("quick", df["transcription"].values)  # Verify transcription is correct
        logger.info("The transcription 'quick' is present in the DataFrame.")

if __name__ == "__main__":
    unittest.main()
