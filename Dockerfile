FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Set the entrypoint
ENTRYPOINT ["python", "src/api/app.py"]

# Expose the port the app runs on
EXPOSE 5000
