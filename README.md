## Project Setup Instructions

To set up the project environment, please follow the steps outlined below:

### 1. Create a Virtual Environment

Creating a virtual environment helps to isolate your project's dependencies and avoid conflicts with other projects. You can create a virtual environment by executing the following command:

    virtualenv .venv

### 2. Activate the Virtual Environment

Once the virtual environment is created, activate it with the following command:

    source .venv/bin/activate

After activation, your shell prompt will change to indicate that the virtual environment is active.

### 3. Install Required Packages

#### Data Handling
- **`pandas`**: Provides powerful data structures like DataFrames for reading, processing, and analyzing tabular data (e.g., CSV files).
- **`numpy`**: Enables efficient numerical operations and array processing, often used for handling image and numerical data.
- **`tqdm`**: A lightweight library to add progress bars, useful for visualizing the execution of loops in scripts.

#### Natural Language Processing
- **`nltk`**: Used for text processing tasks, such as removing stopwords and cleaning transcriptions. Essential for preparing labels in text datasets.

#### Image Processing
- **`opencv-python-headless`**: A library for advanced image processing tasks, such as resizing, transformations, and feature extraction. The "headless" version is used in environments without a GUI.
- **`Pillow`**: A lightweight library for handling and manipulating image files, such as reading, resizing, and converting images.

#### Deep Learning Frameworks
- **`tensorflow`**: A comprehensive deep learning framework used for building and training neural networks. It's the core framework for training the CNN model in this project.
- **`keras`**: An abstraction layer over TensorFlow, simplifying model creation and training workflows.

#### Machine Learning Utilities
- **`scikit-learn`**: Provides tools for machine learning tasks like preprocessing (e.g., label encoding), model evaluation, and train-test splitting.

#### Version Control and Repository Management
- **`dvc`**: A data version control system that manages datasets, models, and ML pipelines effectively. It integrates well with Git and other tools.
- **`dvc-http`**: A DVC extension for enabling HTTP/HTTPS remote storage, useful for connecting repositories with cloud platforms.
- **`dagshub`**: A collaborative platform that integrates with DVC and Git, streamlining ML project versioning and visualization.

#### Security and Cryptography
- **`cryptography`**: A library for secure operations, required by DVC to handle encrypted remote connections securely.

#### Installation

To install the necessary libraries, use the following command:

pip install -r requirements.txt

### 4. Set Up DVC (Data Version Control)

Before pulling the dataset, you need to configure DVC to connect to the remote storage. Follow these steps:

#### a. Add Remote Storage

Add the DVC remote storage by running the command below.

    dvc remote add origin https://dagshub.com/KazemZh/OCR_Handwritting_MLOps.dvc

You can verify your remote setup with:

    dvc remote list

Set the default remote and check its status:

    dvc remote default origin
    dvc remote default

#### b. Configure Authentication

Set DVC remote storage authentication using the following commands. Replace `<your_username>` and `<your_token>` with your actual DagsHub username and API token:

    dvc remote modify origin --local auth basic 
    dvc remote modify origin --local user <your_username>
    dvc remote modify origin --local password <your_token>

Hint: You can find your Setup credentials by navigating to the project DagsHub repository: go to Remote -> Data -> DVC -> HTTP -> Setup credentials.

### 5. Pull the Dataset

With DVC configured, you can now pull the dataset from the remote storage. Execute the following command:

    dvc pull

This command will download the dataset files specified in your DVC configuration.

### 6. Extract the Dataset

    python src/data/extract_raw_data.py 

### 7. Load the dataset

    python src/data/load_dataset.py 

---

Following these steps will set up your project environment correctly and ensure that all necessary dependencies and datasets are available for development and testing. If you encounter any issues, please refer to the troubleshooting section or contact the project maintainers for assistance.