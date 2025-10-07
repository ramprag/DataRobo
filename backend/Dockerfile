# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including build tools for PyTorch and SDV
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# First install torch with CPU support to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Verify GAN libraries are installed
RUN python -c "from sdv.single_table import CTGANSynthesizer; print('✅ SDV/GAN libraries installed successfully')" || \
    echo "⚠️ Warning: GAN libraries not available, will use statistical method"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads synthetic logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]