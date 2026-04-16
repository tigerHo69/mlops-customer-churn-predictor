# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Entrypoint script to run both (using a simple shell script for demo)
RUN echo "#!/bin/sh\npython src/api.py & streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0" > /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
