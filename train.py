import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pickle

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv("Pune_House_Data.csv")

# Remove missing values
df = df.dropna()

# ---------------------------------------------------------
# 2. Split features & target
# ---------------------------------------------------------
X = df.drop("price", axis=1)
y = df["price"]

# Detect column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols   = X.select_dtypes(include=['int64','float64']).columns.tolist()

# ---------------------------------------------------------
# 3. Preprocessing pipeline
# ---------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# ---------------------------------------------------------
# 4. Build Model Pipeline
# ---------------------------------------------------------
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

# ---------------------------------------------------------
# 5. Train-test split
# ---------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model.fit(X_train, Y_train)

# ---------------------------------------------------------
# 6. Save model as model.pkl
# ---------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Training Complete — model.pkl saved successfully!")
