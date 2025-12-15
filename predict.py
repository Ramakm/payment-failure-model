import argparse
import joblib
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser(description="Predict payment failure risk.")
    parser.add_argument("--occupation", type=str, default="worker")
    parser.add_argument("--purpose", type=str, default="bills")
    parser.add_argument("--source", type=str, default="Cash")
    parser.add_argument("--country", type=str, default="US")
    parser.add_argument("--nationality", type=str, default="US")
    parser.add_argument("--receiver_country", type=str, default="US")
    parser.add_argument("--age", type=int, default=30)
    parser.add_argument("--id_verified", type=int, default=1)
    parser.add_argument("--cross_border", type=int, default=0)
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = joblib.load("payment_failure_model.pkl")
    except FileNotFoundError:
        print("Model file not found. Please run train.py first.")
        return

    # Prepare input
    data = {
        "occupation": [args.occupation],
        "purposeOfTransaction": [args.purpose],
        "sourceOfFunds": [args.source],
        "countryOfBirth": [args.country],
        "nationality": [args.nationality],
        "receiver.address.countryCode": [args.receiver_country],
        "age": [args.age],
        "id_verified": [args.id_verified],
        "cross_border": [args.cross_border]
    }
    
    df = pd.DataFrame(data)
    
    # Predict
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    
    result = {
        "payment_failed": int(prediction),
        "failure_probability": float(proba)
    }
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
