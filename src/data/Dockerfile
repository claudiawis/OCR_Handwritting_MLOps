# check nodes and cpus capacites for partition hims as example:
    sinfo -p hims -o "%N %c" --Node | awk 'NR>1 {nodes+=1; cpus+=$2} END {print "Total Nodes:", nodes, "\nTotal CPUs:", cpus}'


    # Use the official Python base image
    FROM python:3.11
    
    # Set the working directory inside the container
    WORKDIR /app
    
    # Copy the Python scripts from src/data
    COPY src/data/*.py ./src/data/
    
    # Copy the requirements file
    COPY src/data/requirements.txt ./requirements.txt
    
    # Copy necessary files from the root directory
    COPY dvc.yaml ./
    COPY dvc.lock ./          
    COPY .gitignore ./        
    COPY .dvcignore ./        
    
    
    # Install the required Python packages
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Run the entire ingestion pipeline up to one_hot_encode_labels
    CMD ["dvc", "repro", "one_hot_encode_labels"]