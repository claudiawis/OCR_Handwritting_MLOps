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
│   │   ├── `Dockerfile`                 <- Dockerfile for the data ingestion pipeline<br>
│   │   └── `requirements.txt`           <- Dependencies required for running the ingestion service<br>
│   │<br>
│   ├── **models/**<br>
│   │   ├── `setup_callbacks.py`         <- Defines training callbacks<br>
│   │   ├── `build_train_cnn.py`         <- Builds and trains the CNN model<br>
│   │   ├── `evaluate_model.py`          <- Evaluates model performance<br>
│   │   ├── `Dockerfile`                 <- Dockerfile for model training and inference pipeline<br>
│   │   └── `requirements.txt`           <- Dependencies required for running the training service<br>
│   │<br>
│   ├── **api/**                         <- Scripts for prediction microservice and FastAPI endpoints<br>
│   │   ├── `api.py`                     <- Loads model and API endpoints for prediction and retraining<br>
│   │   ├── `Dockerfile`                 <- Dockerfile for the prediction microservice<br>
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
│   └── **test_models/**                 <- Unit test scripts for the model training service<br>
│<br>
├── **docs/**                            <- Documentation for the project<br>
│<br>
├── **logs/**                            <- Storing application runtime logs<br>
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

## The App

## Grafana for Monitor

## All Ports