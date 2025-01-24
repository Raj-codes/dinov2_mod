# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements_eval_windows.txt

# Expose??
EXPOSE 5000

# Run your application
CMD ["python", "-m", "dinov2_mod", "--config_path", "config.json"]