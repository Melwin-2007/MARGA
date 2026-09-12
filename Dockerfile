FROM python:3.10-slim

WORKDIR /app

# Install build dependencies if needed for some ML libraries
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the required SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code and models
COPY . .

# Expose the port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
