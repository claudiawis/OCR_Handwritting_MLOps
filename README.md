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

#### Add Remote Storage

Add the DVC remote storage by running the command below.

    dvc remote add origin https://dagshub.com/KazemZh/OCR_Handwritting_MLOps.dvc

You can verify your remote setup with:

    dvc remote list

Set the default remote and check its status:

    dvc remote default origin
    dvc remote default

#### Configure Authentication

Set DVC remote storage authentication using the following commands. Replace `<your_username>` and `<your_token>` with your actual DagsHub username and API token:

    dvc remote modify origin --local auth basic 
    dvc remote modify origin --local user <your_username>
    dvc remote modify origin --local password <your_token>

Hint: You can find your Setup credentials by navigating to the project DagsHub repository: go to Remote -> Data -> DVC -> HTTP -> Setup credentials.

### 5. Pull the Dataset

With DVC configured, you can now pull the dataset from the remote storage. Execute the following command:

    dvc pull

This command will download the dataset files specified in your DVC configuration.

### 6. Dataset Preparation

This section outlines the steps required to prepare the dataset for model training.

#### 1. Extract Raw Data

Extracts the raw data from the source and saves it in a suitable format for further processing.

    python src/data/extract_raw_data.py 

#### 2. Load Dataset

Loads the extracted dataset into memory for processing.

    python src/data/load_dataset.py 

#### 3. Filter Dataset

Applies filters to the dataset to remove unnecessary or irrelevant information.

    python src/data/filter_data.py

#### 4. Clean Dataset

Cleans the dataset by handling missing values, removing duplicates, and correcting inconsistencies.

    python src/data/clean_data.py

#### 5. Encode Dataset

Encodes categorical features into numerical format to prepare them for model training.

    python src/data/encode_data.py

#### 6. Prepare Input Features and Labels

Prepares the input features and target labels for model training.

    python src/data/prepare_features.py 

#### 7. Split the Data into Training and Testing Sets

Divides the dataset into training and testing sets to evaluate model performance.

    python src/data/split_data.py 

This section describes the steps involved in building, training, and evaluating the model.

#### 8. Reshape Data for CNN Input

Reshapes the dataset to fit the input requirements of the Convolutional Neural Network (CNN).

    python src/data/reshape_data.py

#### 9. Calculate Class Weights for Imbalance

Calculates class weights to address potential class imbalance in the dataset.

    python src/data/calculate_class_weights.py

#### 10. One-Hot Encode Labels

Applies one-hot encoding to the target labels to prepare them for multi-class classification.

    python src/data/one_hot_encode_labels.py

### 7. Model Building, Training, and Testing

#### 1. Set Up Callbacks for Training

Configures `early stopping` and `model checkpoint` callbacks to optimize training.

    python src/models/setup_callbacks.py

#### 2. Build, Train, and Save the CNN Model

Constructs the CNN architecture, trains the model on the training data, and saves the trained model.

    python src/models/build_train_cnn.py

#### 3. Evaluation of the CNN Model

Evaluates the performance of the trained CNN model on the test dataset.

    python src/models/evaluate_model.py

### 8. Use the FastAPI Inference API

After training your model, you can deploy an inference API using **FastAPI**. This API allows users to upload images of handwritten words and get predictions.

The FastAPI application is located in the `src/api/api.py` file. You can run the API server using **uvicorn**, a lightweight ASGI server. Use the following command from the main directory:

    uvicorn src.api.api:app --reload

Then open your browser at http://localhost:8000/docs to access the FastAPI interactive docs.

### 9. Airflow

Airflow is used to orchestrate Docker containers in an OCR project to ensure scalable, automated, and reliable execution of OCR tasks. By utilizing Airflow’s Directed Acyclic Graphs (DAGs), we can define, schedule, and monitor workflows that deploy and manage Docker containers for each step in the OCR process. This allows for seamless handling of large datasets, parallel processing, error handling, and efficient resource management, ultimately improving the performance and maintainability of the OCR pipeline.

To run airflow and essentially run the project follow these steps if you already have airflow for docker installed and initialized:

    docker-compose -f docker-compose_airflow.yaml up -d

You can check whether the containers are working fine via:

docker-compose -f docker-compose_airflow.yaml ps

Once running you can reach it here:
http://localhost:8081/

username: airflow
password: airflow


## 10. Authenticate App

We decided to implemente a separate FastAPI application to verify users and authorize their roles. The roles include user and admin

DRAFT - DISREGARD ((user details: name-user1 password-password1
admin details: admin1 password2

cd src/authenticate_app
You can open the verification API via:
xxx

Draft 2:

from the cd /src/authenticate_app

uvicorn fast_api_basic_security:app --reload --host 127.0.0.1 --port 8111

go to: localhost:81111/user in the browser of your choice and enter the user credentials and you will be redirected to the service.1

Then authenticate as user1 with password 1resu 

You should be forwarded directly to the page (need to ensure the prediction service is running before)

---

## Dockerfiles

### Dockerfile for ingestion step:

Make sure that your Docker Daemon (engine) is not running, for , on Windows, Docker relies on Docker Desktop to run the Docker Daemon. So make sure it is already lunched by opening it.
Build the Docker image by running the following command from the root directory:

    docker build -t ingestion_image -f src/data/Dockerfile .

Then run the Docker container with the volume containing the raw data:

    docker run -v "$(pwd)/data/raw:/app/data/raw" -v "$(pwd)/.dvc:/app/.dvc" -v "$(pwd)/.git:/app/.git" ingestion_image

### Dockerfile for training step:

First, create the image for the training step from the root directory using the command bellow:

    docker build -t training_image -f src/models/Dockerfile .

Then run the Docker container that include the required volumes:

    docker run -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/.dvc:/app/.dvc" -v "$(pwd)/.git:/app/.git" training_image

### Dockerfile for prediction step:

Create the image for the prediction step from the root directory using the command bellow:

    docker build -t prediction_image -f src/api/Dockerfile .


Run the Docker container with the command:

    docker run -p 8000:8000 prediction_image

Open your browser at http://localhost:8000/docs to access the FastAPI interactive docs.

### Dockerfile orchestration:

The docker-compose.yml file that will orchestrate your three services (ingestion, training, and prediction), with shared volumes and ports configured accordingly.

Build all services:

    docker-compose build

Run all services:

    docker-compose up

Run services in detached mode (run the services in the background):

    docker-compose up -d

Check Logs for Each Service:

    docker logs ingestion_service
    docker logs training_service
    docker logs prediction_service

To stop the running services:

    docker-compose down


## Monitoring with Prometheus and Grafana

This project includes monitoring capabilities using **Prometheus** for metrics collection and **Grafana** for visualization. Follow the instructions below to set up and use the monitoring features.

### Activating Monitoring
1. Ensure that Prometheus and Grafana services are defined in the `docker-compose.yml` file.
2. Start all services using:
   ```bash
   docker-compose up -d
3. Verify that all services, including Prometheus and Grafana, are running:
   ```bash
   docker ps

### Accessing Prometheus

- Prometheus is available at **[http://localhost:9090](http://localhost:9090)** (or replace `localhost` with your VM's IP address if running on a remote server).  
- Use Prometheus to inspect raw metrics collected from the API.

---

### Setting Up Grafana

- Access Grafana at **[http://localhost:3000](http://localhost:3000)** (or replace `localhost` with your VM's IP address).  
- The default credentials for Grafana are:  
  - **Username:** `admin`  
  - **Password:** `admin` (you may be prompted to change it on the first login).  

---

### Configuring Prometheus as a Data Source in Grafana

1. Navigate to **Connections > Data sources** in the Grafana sidebar.  
2. Choose **Prometheus** as the data source.  
3. Set the connection URL to:  **[http://localhost:9090](http://localhost:9090)** if using Docker Compose.
4. Click **Save & test** to confirm the connection.

### Creating Dashboards in Grafana

1. Navigate to the **Dashboards** tab and click **New > New dashboard**.
2. Click **Add visualization** and select **Prometheus** as the data source.

#### Example Metrics to Monitor

**1. Number of Predictions**  
   - Use the metric:
     ```
     http_requests_total
     ```
   - Apply the label filter:
     ```
     handler = /predict/
     ```
   - This will track successful requests (responses between 200-299) and client-side error requests (responses between 400-499).

**2. Inference Time**  
   - Add the following query to monitor the average inference time over the last 5 minutes:
     ```promql
     rate(inference_time_seconds_sum[5m]) / rate(inference_time_seconds_count[5m])
     ```


---


Following these steps will set up your project environment correctly and ensure that all necessary dependencies and datasets are available for development and testing. If you encounter any issues, please refer to the troubleshooting section or contact the project maintainers for assistance.