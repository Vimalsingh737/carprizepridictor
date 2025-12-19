from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature names
data = joblib.load("model/car_price_model.pkl")
model = data["model"]
feature_names = data["features"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Get form data
            Year = int(request.form["Year"])
            Kms_Driven = float(request.form["Kms_Driven"])
            Owner = int(request.form["Owner"])
            Fuel_Type = request.form["Fuel_Type"]
            Seller_Type = request.form["Seller_Type"]
            Transmission = request.form["Transmission"]
            Mileage = float(request.form["Mileage"])
            Engine = float(request.form["Engine"])
            Power = float(request.form["Power"])

            # Compute Car_Age
            Car_Age = 2025 - Year

            # Feature dict
            features_dict = {
                "Kms_Driven": Kms_Driven,
                "Owner": Owner,
                "Mileage": Mileage,
                "Engine": Engine,
                "Power": Power,
                "Car_Age": Car_Age,
                "Fuel_Type_Diesel": 1 if Fuel_Type=="Diesel" else 0,
                "Fuel_Type_Petrol": 1 if Fuel_Type=="Petrol" else 0,
                "Seller_Type_Individual": 1 if Seller_Type=="Individual" else 0,
                "Transmission_Manual": 1 if Transmission=="Manual" else 0
            }

            # Fill missing features with 0
            for col in feature_names:
                if col not in features_dict:
                    features_dict[col] = 0

            # Convert to DataFrame
            input_df = pd.DataFrame([features_dict], columns=feature_names)

            # Predict
            prediction = model.predict(input_df)[0]
            prediction = round(prediction, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
