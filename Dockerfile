# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Run the command to install dependencies (if any)
# RUN pip install -r requirements.txt

# Command to run the Python script
CMD ["python", "app.py"]
