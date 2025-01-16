from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import os

my_dag = DAG(
    dag_id='docker_orchestration_dag',
    description='Run Docker Containers for Ingestion, Training, and Prediction',
    schedule_interval=None,  
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(2),
    },
    catchup=False,
)

# Task to run the ingestion container
ingestion_task = DockerOperator(
    task_id='run_ingestion',
    image='ingestion_image',  #Kazem's image name
    command="python src/data/ingestion_script.py",  # entry point?
    volumes=[
        f"{os.getcwd()}/data/raw:/app/data/raw",
        f"{os.getcwd()}/.dvc:/app/.dvc",
        f"{os.getcwd()}/.git:/app/.git"
    ],
    dag=my_dag,
)

# Task to run the training container
training_task = DockerOperator(
    task_id='run_training',
    image='training_image',  # Kazem's image name
    command="python src/models/train_model.py",  # entry point?
    volumes=[
        f"{os.getcwd()}/data:/app/data",
        f"{os.getcwd()}/models:/app/models",
        f"{os.getcwd()}/.dvc:/app/.dvc",
        f"{os.getcwd()}/.git:/app/.git"
    ],
    dag=my_dag,
)

# Task to run the prediction container
prediction_task = DockerOperator(
    task_id='run_prediction',
    image='prediction_image',  # Kazem's image name
    command="uvicorn src/api.main:app --host 0.0.0.0 --port 8000",  # entry point?
    ports=[8000],
    dag=my_dag,
)

# Define the task dependencies (run sequentially in this case)
ingestion_task >> training_task >> prediction_task

