import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/clean_2017-2018-2019-2020-2021-2022-playerstats.csv')

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

# Split the weighted data into training and test sets
train_data_weighted = data_sorted[data_sorted["Year"] < 2022]
test_data_weighted = data_sorted[data_sorted["Year"] == 2022]

# Define the features and target for the weighted data
X_train_weighted = train_data_weighted.drop(columns=["PPR"])
y_train_weighted = train_data_weighted["PPR"]

X_test_weighted = test_data_weighted.drop(columns=["PPR"])
y_test_weighted = test_data_weighted["PPR"]

# Apply preprocessor to the weighted training and test sets
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(), categorical_columns)])

X_train_weighted = preprocessor.fit_transform(X_train_weighted)
X_test_weighted = preprocessor.transform(X_test_weighted)

# Train the Gradient Boosting model on the weighted training data
gb_model_weighted = GradientBoostingRegressor(random_state=42)
gb_model_weighted.fit(X_train_weighted, y_train_weighted)

# Make predictions on the weighted test data
y_pred_weighted = gb_model_weighted.predict(X_test_weighted)

# Calculate the RMSE of the predictions on the weighted test data
rmse_weighted = np.sqrt(mean_squared_error(y_test_weighted, y_pred_weighted))

# Create a copy of the data for inverse transformation
data_label_encoded = data_sorted.copy()

# Calculate the projected PPR values for 2022
projected_ppr_2022 = pd.DataFrame()
projected_ppr_2022["Player"] = le.inverse_transform(data_label_encoded.loc[test_data_weighted.index, "Player"].values)
projected_ppr_2022["PPR_Projected"] = y_pred_weighted

# Extract the actual PPR values for 2022
actual_ppr_2022 = pd.DataFrame()
actual_ppr_2022["Player"] = le.inverse_transform(data_label_encoded.loc[test_data_weighted.index, "Player"])
actual_ppr_2022["PPR_Actual"] = y_test_weighted.values

# Merge the actual and projected PPR dataframes
ppr_comparison = pd.merge(actual_ppr_2022, projected_ppr_2022, on="Player")

# Calculate the error for each player
ppr_comparison["Error"] = ppr_comparison["PPR_Projected"] - ppr_comparison["PPR_Actual"]

# Calculate the RMSE
rmse_comparison = np.sqrt(mean_squared_error(ppr_comparison["PPR_Actual"], ppr_comparison["PPR_Projected"]))

# Display the RMSE and the comparison dataframe
print("RMSE:", rmse_comparison)
print(ppr_comparison.head())

#Export test csv
ppr_comparison.sort_values('PPR_Actual', ascending = False).to_csv('export/2022-test-results')
