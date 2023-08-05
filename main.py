import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the data from the CSV file
data = pd.read_csv('data/clean_2010-2011-2012-2013-2014-2015-2016-2017-2018-2019-2020-2021-2022-playerstats.csv')

# Drop the unnecessary column
data = data.drop(columns=["Unnamed: 0"])

# Encode categorical columns
categorical_columns = ["FantPos", "Tm"]
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns.remove("PPR")  # Remove "PPR" from numeric columns

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to "Player" column
data["Player"] = le.fit_transform(data["Player"])

# Initialize the decay factor
decay_factor = 0.9

# Sort the data by player and year
data_sorted = data.sort_values(["Player", "Year"])

# Calculate the weights for each row
weights = decay_factor ** (data_sorted.groupby("Player").cumcount(ascending=False))

# Apply the weights to the PPR values and numerical features
data_sorted["PPR"] *= weights
for column in numeric_columns:
    data_sorted[column] *= weights

# Keep only the last occurrence of each player's data
data_last_occurrence = data_sorted.drop_duplicates(subset="Player", keep="last")

# Use all the data for training (no test data for 2023)
train_data_weighted = data_last_occurrence

# Define the features and target for the weighted data
X_train_weighted = train_data_weighted.drop(columns=["PPR"])
y_train_weighted = train_data_weighted["PPR"]

# Apply preprocessor to the weighted training set
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(), categorical_columns)])

X_train_weighted = preprocessor.fit_transform(X_train_weighted)

# Train the Gradient Boosting model on the weighted training data
gb_model_weighted = GradientBoostingRegressor(random_state=42)
gb_model_weighted.fit(X_train_weighted, y_train_weighted)

# Predict PPR for 2023
predicted_ppr_2023 = gb_model_weighted.predict(X_train_weighted)

# Create a copy of the data for inverse transformation
data_label_encoded = data_last_occurrence.copy()

# Calculate the projected PPR values for 2023
projected_ppr_2023 = pd.DataFrame()
projected_ppr_2023["Player"] = le.inverse_transform(data_label_encoded["Player"].values)
projected_ppr_2023["PPR_Projected_2023"] = predicted_ppr_2023

# Display the projected PPR values for 2023
print(projected_ppr_2023)

# Export CSV
projected_ppr_2023.sort_values('PPR_Projected_2023', ascending = False).to_csv('export/2023-projections.csv')
