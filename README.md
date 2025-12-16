# Payment Failure Model

## Overview
This repository contains a simple machine‑learning pipeline that predicts whether a payment transaction will fail based on customer KYC information and transaction metadata. The model is a logistic regression trained on synthetic data that mimics real‑world payment onboarding records.

## Project Structure
- `userdata.json` – Synthetic dataset used for training and evaluation.
- `train.py` – Script that loads the data, performs feature engineering, trains a logistic regression model, logs the run with MLflow, and saves the model to `payment_failure_model.pkl`.
- `predict.py` – Script that loads the saved model and makes a prediction for a single example.
- `app.py` – FastAPI application exposing a `/predict` endpoint that returns the failure probability.
- `requirements.txt` – Python dependencies required to run the project.
- `README.md` – This documentation.

## Setup
1. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   The required packages include `pandas`, `scikit‑learn`, `mlflow`, `fastapi`, and `uvicorn`.
3. **Verify the environment**
   ```bash
   python -c "import pandas, sklearn, mlflow; print('All imports succeeded')"
   ```

## Training the Model
Run the training script:
```bash
python train.py
```
The script will:
- Load `userdata.json`.
- Engineer features such as age, ID verification flag, and cross‑border indicator.
- Split the data into training and test sets.
- Train a logistic regression model.
- Log parameters, metrics, and the model artifact to an MLflow tracking server (default local SQLite store).
- Save the trained model to `payment_failure_model.pkl`.

After successful execution you will see a classification report and a confirmation that the model file has been saved.

## Making a Prediction (CLI)
Use the `predict.py` script to obtain a prediction for a single record:
```bash
python predict.py
```
The output is a JSON object, for example:
```json
{ "payment_failed": 1, "failure_probability": 0.62 }
```

## Predicting on New Data
You can use the same `predict.py` script to make predictions on any new data record. Provide a JSON object that matches the feature schema used during training. For batch predictions, create a JSON Lines file where each line is a separate record and modify `predict.py` to read the file and output predictions.

### Example JSON for a single record
```json
{
  "occupation": "worker",
  "purposeOfTransaction": "shopping",
  "sourceOfFunds": "Cash",
  "countryOfBirth": "IN",
  "nationality": "IN",
  "idVerificationStatus": "N",
  "receiver": {"address": {"countryCode": "US"}},
  "dateOfBirth": "1990-01-01"
}
```

### Using the script
```bash
python predict.py '{"occupation":"worker","purposeOfTransaction":"shopping","sourceOfFunds":"Cash","countryOfBirth":"IN","nationality":"IN","idVerificationStatus":"N","receiver":{"address":{"countryCode":"US"}},"dateOfBirth":"1990-01-01"}'
```

The script will print a JSON response with `payment_failed` and `failure_probability`.

For batch predictions, create a file `new_data.jsonl` with one JSON per line and run:

```bash
python batch_predict.py new_data.jsonl
```
(You can create `batch_predict.py` by copying `predict.py` and looping over each line.)

You can also send the same JSON payload to the FastAPI endpoint as shown in the FastAPI section.


## FastAPI Service
The FastAPI wrapper (`app.py`) provides a REST endpoint:
- **Start the server**
  ```bash
  uvicorn app:app --reload
  ```
- **POST request** to `http://127.0.0.1:8000/predict` with a JSON payload matching the feature schema. Example using `curl`:
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"occupation":"worker","purposeOfTransaction":"shopping","sourceOfFunds":"Cash","countryOfBirth":"IN","nationality":"IN","idVerificationStatus":"N","receiver":{"address":{"countryCode":"US"}},"dateOfBirth":"1990-01-01"}'
  ```
  The response will contain `payment_failure_risk` (0 or 1) and `failure_probability`.

## MLflow Tracking UI
To view the logged runs, start the MLflow UI:
```bash
mlflow ui
```
If you encounter an import error related to `alembic`, ensure that the virtual environment is activated and that the `alembic` package version is compatible with your Python version. Re‑installing the dependencies often resolves the issue:
```bash
pip install --upgrade alembic
```
The UI is available at `http://127.0.0.1:5000`.

## Clean‑up
To deactivate the virtual environment:
```bash
deactivate
```
You can delete the `venv` directory and the generated model file (`payment_failure_model.pkl`) if you wish to start fresh.

---
*This README provides a reproducible workflow for training, evaluating, and serving a payment‑failure prediction model with MLflow tracking and FastAPI.*
## Streamlit Deployment

A live demo of the model is available at https://payment-failure.streamlit.app/. You can interact with the UI to upload a JSON file for batch predictions or fill in the form for a single prediction. The Streamlit app uses the same model and feature engineering logic as the FastAPI service.

## Build by

Ramakrushna Mohapatra 
- [LinkedIn](https://www.linkedin.com/in/ramakrushnamohapatra/)
- [GitHub](https://github.com/Ramakm)
- [X](https://x.com/codewith_ram)
- [Medium](https://medium.com/@techwith.ram)
- [Instagram](https://www.instagram.com/techwith.ram/)