# AnomalyNodeML

This service focuses on server log data to detect anomalies and predict them accurately using a TensorFlow model.

### Steps to Build the Model:

1. **Data Processing**:
   - Data Cleaning
   - Removing Unnecessary Columns
   - Transforming Categorical Data using Label Encoder
2. **Model Building**:
   - A Sequential TensorFlow Model
3. **Model Evaluation**:
   - Evaluate the Model on Test Data

The model can be trained with multiple training datasets and evaluated against evaluation data.

### Running the Service Locally Using Docker:

1. Build the Docker image and run the service:
   ```bash
   docker-compose build --no-cache
   docker-compose up -d
2. To stop and remove the service:
    ```bash
    docker-compose down