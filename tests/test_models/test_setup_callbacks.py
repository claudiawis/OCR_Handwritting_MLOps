import unittest
import os
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.setup_callbacks import setup_callbacks

class TestSetupCallbacks(unittest.TestCase):

    def setUp(self):
        # Set up temporary file paths
        self.callbacks_path = "tests/test_models/tmp_callbacks.keras"
        self.model_checkpoint_path = "tests/test_models/tmp_best_model.keras"

    def tearDown(self):
        # Remove temporary files
        for path in [self.callbacks_path, self.model_checkpoint_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_setup_callbacks(self):
        # Call the setup_callbacks function
        callbacks = setup_callbacks(self.callbacks_path, self.model_checkpoint_path)

        # Verify the callbacks list contains EarlyStopping and ModelCheckpoint
        self.assertTrue(any(isinstance(cb, EarlyStopping) for cb in callbacks))
        self.assertTrue(any(isinstance(cb, ModelCheckpoint) for cb in callbacks))

        # Verify the callbacks file is created
        self.assertTrue(os.path.exists(self.callbacks_path))

        # Load the saved callbacks
        with open(self.callbacks_path, "rb") as f:
            saved_callbacks = pickle.load(f)

        # Verify the loaded callbacks
        self.assertTrue(any(isinstance(cb, EarlyStopping) for cb in saved_callbacks))
        self.assertTrue(any(isinstance(cb, ModelCheckpoint) for cb in saved_callbacks))

if __name__ == "__main__":
    unittest.main()
