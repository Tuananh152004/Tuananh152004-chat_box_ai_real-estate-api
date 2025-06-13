FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY app.py .
COPY real_estate_combined_index.faiss real_estate_combined_text.csv ./

# Expose port for Cloud Run or local testing
EXPOSE 8080

# Start application with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2"]
