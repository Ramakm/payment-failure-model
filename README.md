# payment-failure-model

Simple machine learning model that predicts whether a payment transaction may fail based on customer KYC and transaction metadata.

This project demonstrates a simple machine learning model that predicts whether a payment transaction may fail based on customer KYC and transaction metadata.

The project uses synthetic data that mimics real-world payment onboarding inputs.

## Data
- Format: JSON
- Records: 50 users
- Data includes:
  - KYC details
  - Occupation
  - Source of funds
  - Transaction purpose
  - Receiver country

## Model
- Algorithm: Logistic Regression
- Reason:
  - Simple
  - Explainable
  - Suitable for compliance-heavy environments

## Features Used
- Age
- ID verification status
- Cross-border indicator
- Occupation
- Source of funds
- Transaction purpose

## Target
- payment_failed (0 or 1)

The label is generated using deterministic business rules
to simulate real failure patterns.

## Output
- Trained model saved as `payment_failure_model.pkl`
- Evaluation metrics printed in notebook
- Feature importance available for explainability
