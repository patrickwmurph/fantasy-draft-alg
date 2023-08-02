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

# Data Cleaning
data = data.drop(columns=["Unnamed: 0"])
categorical_columns = ["FantPos", "Tm"]
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns.remove("PPR")  

# Int Encoder
le = LabelEncoder()

data["Player"] = le.fit_transform(data["Player"])

#Create and Apply weights for each year
decay_factor = 0.9

data_sorted = data.sort_values(["Player", "Year"])

weights = decay_factor ** (data_sorted.groupby("Player").cumcount(ascending=False))

data_sorted["PPR"] *= weights
for column in numeric_columns:
    data_sorted[column] *= weights

# Training and test sets
train_data_weighted = data_sorted[data_sorted["Year"] < 2022]
test_data_weighted = data_sorted[data_sorted["Year"] == 2022]

X_train_weighted = train_data_weighted.drop(columns=["PPR"])
y_train_weighted = train_data_weighted["PPR"]

X_test_weighted = test_data_weighted.drop(columns=["PPR"])
y_test_weighted = test_data_weighted["PPR"]

# Build and Apply Processor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(), categorical_columns)])

X_train_weighted = preprocessor.fit_transform(X_train_weighted)
X_test_weighted = preprocessor.transform(X_test_weighted)

gb_model_weighted = GradientBoostingRegressor(random_state=42)
gb_model_weighted.fit(X_train_weighted, y_train_weighted)

y_pred_weighted = gb_model_weighted.predict(X_test_weighted)

rmse_weighted = np.sqrt(mean_squared_error(y_test_weighted, y_pred_weighted))

preprocessor.fit(data_sorted.drop(columns=["PPR"]))

gb_model_weighted.fit(preprocessor.transform(data_sorted.drop(columns=["PPR"])), data_sorted["PPR"])

# Generate 2023 Data
data_2023_weighted = data_sorted.copy()
data_2023_weighted["Year"] = 2023
data_2023_weighted["PPR"] = gb_model_weighted.predict(preprocessor.transform(data_2023_weighted.drop(columns=["PPR"])))

# Remove duplicated by Averaging Predictions into Fianl Pred
final_predictions_weighted = data_2023_weighted.groupby("Player", as_index=False)["PPR"].mean()
final_predictions_weighted["Player"] = le.inverse_transform(final_predictions_weighted["Player"])

#Top n Players by PPR
final_predictions_weighted.sort_values('PPR', ascending=False).head(n=10)