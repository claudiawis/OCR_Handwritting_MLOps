import unittest
import os
import pickle
import logging
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.setup_callbacks import setup_callbacks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSetupCallbacks(unittest.TestCase):

    def setUp(self):
        # Set up temporary file paths
        self.callbacks_path = "tests/test_models/tmp_callbacks.keras"
        self.model_checkpoint_path = "tests/test_models/tmp_best_model.keras"
        logger.info("Temporary file paths set: callbacks_path=%s, model_checkpoint_path=%s", 
                    self.callbacks_path, self.model_checkpoint_path)

    def tearDown(self):
        # Remove temporary files
        for path in [self.callbacks_path, self.model_checkpoint_path]:
            if os.path.exists(path):
                os.remove(path)
                logger.info("Removed temporary file: %s", path)

    def test_setup_callbacks(self):
        logger.info("Testing setup_callbacks function...")

        # Call the setup_callbacks function
        callbacks = setup_callbacks(self.callbacks_path, self.model_checkpoint_path)

        # Verify the callbacks list contains EarlyStopping and ModelCheckpoint
        self.assertTrue(any(isinstance(cb, EarlyStopping) for cb in callbacks), 
                        "EarlyStopping callback not found.")
        self.assertTrue(any(isinstance(cb, ModelCheckpoint) for cb in callbacks), 
                        "ModelCheckpoint callback not found.")
        logger.info("Callbacks verified: EarlyStopping and ModelCheckpoint are present.")

        # Verify the callbacks file is created
        self.assertTrue(os.path.exists(self.callbacks_path), 
                        f"{self.callbacks_path} was not created.")
        logger.info("Callbacks file created: %s", self.callbacks_path)

        # Load the saved callbacks
        with open(self.callbacks_path, "rb") as f:
            saved_callbacks = pickle.load(f)
            logger.info("Loaded callbacks from file: %s", self.callbacks_path)

        # Verify the loaded callbacks
        self.assertTrue(any(isinstance(cb, EarlyStopping) for cb in saved_callbacks), 
                        "EarlyStopping callback not found in loaded callbacks.")
        self.assertTrue(any(isinstance(cb, ModelCheckpoint) for cb in saved_callbacks), 
                        "ModelCheckpoint callback not found in loaded callbacks.")
        logger.info("Loaded callbacks verified: EarlyStopping and ModelCheckpoint are present.")

if __name__ == "__main__":
    unittest.main()
