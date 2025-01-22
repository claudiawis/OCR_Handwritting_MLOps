# Use the official Python base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the Python scripts from src/data
COPY src/models/*.py ./src/models/

# Copy the requirements file
COPY src/models/requirements.txt ./requirements.txt

# Copy necessary files from the root directory
COPY dvc.yaml ./
COPY dvc.lock ./          
COPY .gitignore ./        
COPY .dvcignore ./
COPY metrics/ ./metrics/        


# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Run the entire ingestion pipeline up to one_hot_encode_labels
CMD ["sh", "-c", "dvc repro --downstream setup_callbacks"]
