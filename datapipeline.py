import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# --- Step 1: Extract ---
df = pd.read_csv("sales.csv")

# --- Step 2: Transform ---

# Define feature types
numeric_features = ['Units Sold', 'Unit Price']
categorical_features = ['Category']

# Numeric preprocessing pipeline
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing pipeline
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Apply transformation
processed_data = preprocessor.fit_transform(df)

# Convert to DataFrame
# Get column names from transformers
encoded_columns = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
all_columns = numeric_features + list(encoded_columns)
processed_df = pd.DataFrame(processed_data.toarray() if hasattr(processed_data, 'toarray') else processed_data,
                            columns=all_columns)

# --- Step 3: Load ---
processed_df.to_csv("processed_sales_data.csv", index=False)
print("✅ Data pipeline completed. Processed data saved to 'processed_sales_data.csv'")
