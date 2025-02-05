# Use a base image with Python
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching
COPY ../requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./src ./src/
COPY ./tests ./tests/       

# Set the command to run the tests
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "*.py"]
