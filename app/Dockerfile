# First Stage: Build Environment
FROM python:3.12 AS builder

# Clear apt cache
RUN rm -rf /var/lib/apt/lists/*

# Install system dependencies including HDF5
RUN set -x \
    && apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libhdf5-dev \
        libhdf5-serial-dev \
        hdf5-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Second Stage: Final Image
FROM python:3.12

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local /usr/local

# Set the working directory
WORKDIR /app

# Copy the 'app/' directory from the host to the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Define the default command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
