# Handwritten Text Recognition Using Deep Learning (OCR MLOps Project)

## Introduction

Handwritten text recognition (HTR) is a crucial step in document digitization, allowing computers to extract and process handwritten text from images. This project aims to develop an end-to-end **Optical Character Recognition (OCR)** system for handwritten documents, leveraging **deep learning techniques** and **MLOps principles** to ensure efficient model deployment and maintenance.

Traditional OCR engines, such as PyTesseract and EasyOCR, perform well on printed text but struggle with handwriting due to variations in styles, spacing, and distortions. To address these challenges, we designed a **custom deep learning model** based on **Convolutional Neural Networks (CNNs)**.

![OCR Example](docs/images/introduction.webp)

This project is designed to integrate **automated pipelines** for:
- **Data preprocessing**
- **Model training and evaluation**
- **Monitoring and logging**
- **Deployment and inference**

The solution is intended for industries such as **insurance, healthcare, and administrative services**, where the digitization of handwritten documents is essential for efficiency and cost reduction.

## Project Organization
  
OCR_Handwriting_MLOps<br>
│<br>
├── **src/**                             <- Source code for the OCR pipeline<br>
│<br>
│   ├── **data/**<br>
│   │   ├── `extract_raw_data.py`        <- Extracts the raw data from the compressed dataset<br>
│   │   ├── `load_dataset.py`            <- Loads the dataset into memory<br>
│   │   ├── `filter_data.py`             <- Filters out unwanted or corrupted data samples<br>
│   │   ├── `clean_data.py`              <- Cleans and preprocesses raw text/image data<br>
│   │   ├── `encode_data.py`             <- Encodes categorical or textual data into numerical format<br>
│   │   ├── `prepare_features.py`        <- Prepares feature vectors for model training<br>
│   │   ├── `split_data.py`              <- Splits the dataset into training and test sets<br>
│   │   ├── `reshape_data.py`            <- Reshapes data to fit the input requirements of the deep learning model<br>
│   │   ├── `calculate_class_weights.py` <- Computes class weights to handle class imbalance in training<br>
│   │   ├── `one_hot_encode_labels.py`   <- Applies one-hot encoding to categorical labels<br>
│   │   ├── `ingestion.py`               <- FastAPI app exposing an '/ingest' endpoint to trigger the DVC pipeline stage for data ingestion.<br>
│   │   ├── `Dockerfile-ingestion`       <- Dockerfile for the data ingestion pipeline<br>
│   │   └── `requirements.txt`           <- Dependencies required for running the ingestion service<br>
│   │<br>
│   ├── **models/**<br>
│   │   ├── `setup_callbacks.py`         <- Defines training callbacks<br>
│   │   ├── `build_train_cnn.py`         <- Builds and trains the CNN model<br>
│   │   ├── `evaluate_model.py`          <- Evaluates model performance<br>
│   │   ├── `training.py`                <- FastAPI app exposing a '/train' endpoint to trigger the training pipeline via DVC.<br>
│   │   ├── `Dockerfile-training`        <- Dockerfile for model training and inference pipeline<br>
│   │   └── `requirements.txt`           <- Dependencies required for running the training service<br>
│   │<br>
│   ├── **api/**                         <- Scripts for prediction microservice and FastAPI endpoints<br>
│   │   ├── `prediction.py`              <- Loads OCR model and define API '/predict' endpoints for prediction<br>
│   │   ├── `gateway.py`                 <- Implements authentication, role-based access control, and request distribution for prediction, training, and ingestion services<br>
│   │   ├── `Dockerfile-prediction`      <- Dockerfile for the prediction microservice<br>
│   │   ├── `Dockerfile-gateway`         <- Dockerfile for the Gateway Service<br>
│   └   └── `requirements.txt`           <- Dependencies required for running the prediction service<br>
│<br>
├── **data/**                            <- Directory for storing raw and processed data<br>
│<br>
│   ├── **processed/**                   <- Processed data<br>
│   │   └── `.gitignore`                 <- Files to be excluded from Git version control dataset<br>
│   │<br>
│   ├── **raw/**                         <- Raw data<br>
│   │   ├── `.gitignore`                 <- Files to be excluded from Git version control dataset<br>
│   │   ├── **raw_data/data/raw/**       <- Extracted data<br>
│   │   │   ├── **words/**               <- Extracted images<br>
│   └   └── **ascii/**                   <- Extracted metadata<br>
│<br>
├── **models/**                          <- Saved trained models<br>
│<br>
├── **prometheus_data/**                 <- Stores Prometheus configuration and monitoring data<br>
│   ├── `alerting_rules`                 <- Defines alerting rules for triggering notifications<br>
│   └── `prometheus.yml`                 <- Prometheus configuration file<br>
│<br>
├── **grafana_data/**                    <- Stores Grafana-related configuration and data<br>
│   ├── **provisioning/**<br>
│   │   ├── **dashboards/**<br>
│   │   │   └── `dashboards.yaml`        <- Specifies available dashboards configuration<br>
│   │   ├── **datasources/**<br>
│   └   └   └── `datasource.yml`         <- Specifies data source configuration<br>
│<br>
├── **alertmanager/**                    <- Configures alerting rules for system failures<br>
│<br>
├── **tests/**                           <- Contains unit test scripts<br>
│   ├── **test_data/**                   <- Unit test scripts for the data ingestion service<br>
│   ├── **test_models/**                 <- Unit test scripts for the model training service<br>
│   └── `Dockerfile-tests`               <- Dockerfile for the test Service<br>
│<br>
├── **docs/**                            <- Documentation for the project<br>
│<br>
├── **logs/**                            <- Storing application runtime logs<br>
│<br>
├── **.dvc/**                            <- stores stores metadata for DVC-tracked files, cache, and configurations<br>
│<br>
├── **.github/**                         <- Files and folders to be excluded from Git version control<br>
│   ├── **workflow/**                    <- Unit test scripts for the data ingestion service<br>
│   └   └── `test.yml`                   <- CI workflow to install dependencies, pull DVC data, check files, and run tests<br>
│<br>
├── `.gitignore`                         <- Files and folders to be excluded from Git version control<br>
├── `.dvcignore`                         <- Files and folders to be excluded from DVC tracking<br>
├── `.dockerignore`                      <- Files and folders to be excluded from Docker builds<br>
├── `docker-compose.yml`                 <- Runs containerized services (API, database, monitoring)<br>
├── `dvc.lock`                           <- DVC metadata tracking file<br>
├── `dvc.yaml`                           <- Defines DVC pipeline stages <br>
├── `run_mlops_pipeline.ah`              <- Automates the pipeline in Linux (prompt command)<br>
├── `run_mlops_pipeline.ps1`             <- Automates the pipeline in Windows (PowerShell)<br>
├── `requirements.txt`                   <- Dependencies required for running the project<br>
├── `TECHNICAL_GUIDE.md`                 <- Detailed technical guide<br>
└── `README.md`                          <- Project overview<br>

## Architecture Diagram

The diagram below illustrates the overall architecture of the project. It consists of multiple Docker services orchestrated together, including ingestion, training, and prediction, all managed under GitHub Actions and DagsHub.  

- **Gateway**: Manages authentication and authorization for all requests.  
- **Ingestion Service**: Handles data collection and preprocessing.  
- **Training Service**: Trains machine learning models using the ingested data.  
- **Prediction Service**: Provides model inference for users.  
- **MLflow**: Tracks experiments and model versions.  
- **Monitoring**: Automate system health checks.
- **Cron Jobs**: Automate retraining and system health checks.   

![Architecture](docs/images/architecture.png)

## API Endpoints:

The **Gateway Service** acts as a central API that routes requests to the **Ingestion**, **Training**, and **Prediction** services. It ensures that only authorized users can access specific endpoints.

To access the Gateway, open your browser and go to: **[http://localhost:8000](http://localhost:8000)**  

![Gateway API Endpoints](docs/images/API2.png)

The table below summarizes the access control for each service:

| Service   | Access Level |  Username | Password |
|-----------|-------------|----------|----------|
| **Prediction** | All Users   |  `user1`  | `1resu`  |
| **Ingestion** | Admins Only  |  `admin1`  | `1nimda`  |
| **Training**  | Admins Only  |  `admin1`  | `1nimda`  |




## Grafana for Monitor

The project includes **Grafana** for real-time monitoring and visualization of system metrics. Grafana is configured to track the performance of the ingestion, training, and prediction services, as well as resource usage (CPU, memory, and network activity).

Below is a screenshot of the Grafana dashboard:

![Monitoring](docs/images/Prometheus_Grafana.png)

To access Grafana, open your browser and go to:**[http://localhost:3000](http://localhost:3000)**

**Default Credentials:**
- **Username:** `admin`
- **Password:** `admin` (Change this after first login!)

## Service Ports Summary

| Service       | Container Name       | Port (Host:Container) |
|--------------|----------------------|-----------------------|
| Gateway      | gateway_service       | 8000:8000            |
| Ingestion    | ingestion_service     | 8100:8100            |
| Training     | training_service      | 8200:8200            |
| Prediction   | prediction_service    | 8300:8300            |
| Prometheus   | prometheus_service    | 9090:9090            |
| Grafana      | grafana_service       | 3000:3000            |

> **Note:** These ports are based on `docker-compose.yml`.
