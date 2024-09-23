# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working dirctory in the container
WORKDIR /app

# Copy the current dirctory contents into the container at /app
COPY . /app

# Install the necessary dependenceies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the application
CMD ["python", "app.py"]
