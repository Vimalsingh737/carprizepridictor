import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/car data.csv")

# Feature engineering
df['Car_Age'] = 2025 - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model + feature names
joblib.dump({
    "model": model,
    "features": X.columns.tolist()
}, "model/car_price_model.pkl")

print("Model trained and saved successfully!")
print("Features:", X.columns.tolist())
