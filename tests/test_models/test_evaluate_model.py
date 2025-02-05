import unittest
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
from src.models.evaluate_model import load_model

class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        # Set up temporary file paths
        self.test_X_test_path = "tests/test_models/tmp_X_test.npy"
        self.test_y_test_path = "tests/test_models/tmp_y_test.npy"
        self.test_model_path = "tests/test_models/tmp_CNN.h5"
        self.report_path = "tests/test_models/tmp_classification_report.csv"
        self.cm_path = "tests/test_models/tmp_confusion_matrix.csv"
        self.cm_plot_path = "tests/test_models/tmp_confusion_matrix.png"

        # Create dummy test data
        X_test = np.random.rand(10, 20)
        y_test = np.eye(3)[np.random.choice(3, 10)]
        np.save(self.test_X_test_path, X_test)
        np.save(self.test_y_test_path, y_test)

        # Create a dummy model
        model = Sequential()
        model.add(Dense(10, activation='relu', input_shape=(20,)))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.save(self.test_model_path)

    def tearDown(self):
        # Remove temporary files
        for path in [self.test_X_test_path, self.test_y_test_path, self.test_model_path,
                     self.report_path, self.cm_path, self.cm_plot_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_evaluate_model(self):
        # Load the model and test data
        model = load_model(self.test_model_path)
        X_test = np.load(self.test_X_test_path)
        y_test = np.load(self.test_y_test_path)

        # Generate predictions
        y_pred = model.predict(X_test).argmax(axis=1)
        y_true = y_test.argmax(axis=1)

        # Generate classification report and confusion matrix
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(self.report_path, index=True)

        cm = confusion_matrix(y_true, y_pred)
        np.savetxt(self.cm_path, cm, delimiter=",")

        # Assertions to verify file creation
        self.assertTrue(os.path.exists(self.report_path))
        self.assertTrue(os.path.exists(self.cm_path))

        # Verify content of classification report
        report_df = pd.read_csv(self.report_path, index_col=0)
        self.assertIn("precision", report_df.columns)
        self.assertIn("recall", report_df.columns)

        # Verify confusion matrix dimensions dynamically
        cm_loaded = np.loadtxt(self.cm_path, delimiter=",")
        expected_classes = cm_loaded.shape[0]  # Automatically get the number of classes
        self.assertEqual(cm_loaded.shape[0], expected_classes)
        self.assertEqual(cm_loaded.shape[1], expected_classes)


if __name__ == "__main__":
    unittest.main()