import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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

# Split the data into training set (before 2022) and test set (2022)
data_train = data_sorted[data_sorted["Year"] < 2022]
data_test = data_sorted[data_sorted["Year"] == 2022]


# Calculate the weights for each row in the training set
weights_train = decay_factor ** (data_train.groupby("Player").cumcount(ascending=False))

# Apply the weights to the PPR values and numerical features in the training set
data_train.loc[:, "PPR"] *= weights_train
for column in numeric_columns:
    data_train.loc[:, column] *= weights_train

# Define the features and target for the training data
X_train = data_train.drop(columns=["PPR"])
y_train = data_train["PPR"]

# Define the features and target for the test data
X_test = data_test.drop(columns=["PPR"])
y_test = data_test["PPR"]

# Apply preprocessor to the training and test sets
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(), categorical_columns)])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train the Gradient Boosting model on the training data
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Predict PPR for 2022
predicted_ppr_2022 = gb_model.predict(X_test)

# Calculate the Mean Squared Error (MSE) between the projected and actual PPR values for 2022
mse_2022 = mean_squared_error(y_test, predicted_ppr_2022)
print(f"MSE for 2022: {mse_2022}")


# Calculate the projected PPR values for 2022
projected_ppr_2022 = pd.DataFrame()
projected_ppr_2022["Player"] = le.inverse_transform(data_test["Player"].values)
projected_ppr_2022["PPR_Projected_2022"] = predicted_ppr_2022
projected_ppr_2022["PPR_Actual_2022"] = data_test["PPR"].values

# Calculate ranks based on projected and actual PPR
projected_ppr_2022["Projected_Rank"] = projected_ppr_2022["PPR_Projected_2022"].rank(ascending=False)
projected_ppr_2022["Actual_Rank"] = projected_ppr_2022["PPR_Actual_2022"].rank(ascending=False)

projected_ppr_2022.sort_values('Actual_Rank').to_csv('export/2022-test-results.csv')



