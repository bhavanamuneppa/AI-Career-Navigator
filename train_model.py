import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("dataset/placement_data.csv")

# Convert Yes/No to 1/0
le = LabelEncoder()
data["Placement"] = le.fit_transform(data["Placement"])

# Features and target
X = data[["CGPA", "Projects", "Internships", "Certifications", "CodingScore"]]
y = data["Placement"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/placement_model.pkl")

print("Model trained successfully!")
print("Saved as model/placement_model.pkl")