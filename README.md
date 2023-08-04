# Machine Learning PPR Projection - 2023

This repository contains a predictive model for forecasting the Player Performance Rating (PPR) of fantasy football players. The model uses historical player statistics from the years 2017 to 2022 to predict the PPR values for the year 2023. The model is based on a Gradient Boosting Regression algorithm and player data from [pro-football-reference](https://www.pro-football-reference.com/). These PPR figures are then used to generate player VORP which can be used to rank players in a drafting order. The results can be found in **2023-draft-order.csv**.


## Authors

- [@patrickwmurph](https://github.com/patrickwmurph)


## Datasets :

The datasets used for training, testing, and deploying the PPR model can be found in the **data** directory. The data contains player statistics for each year, including the player name, position, team, and various performance metrics. The results of the PPR projections for 2023 and model testing on 2022 statistics can be found in the **export** directory titled, **2023-projections.csv** and **2022-test-results.csv** respectively. If you want a description of the model, see [Appendix A: Model Architecture](#appendix).

## Results

The results of this analysis can be found in **2023-draft-order.csv**. Player VORP was calculated assuming the following specifications :

- 12 Team draft
- Each team needs at least : 1 QB, 2 RB, 2 WR, 1 TE

If you want to change these specifications for your specific draft or broaden the range of years to train the model see [Deployment](#local-deployment).

**NOTE** : Players who have been drafted in 2023, or for any other reason have no player data in the past 12 years will not have their PPR projected (i.e. Bijan Robinson).

### Glossary :

- **Rank** : A players relative rank based on VORP
- **PPR_Projected_2023** : A players projected PPR for the coming year based on the model.
- **VORP** : A players VORP (Value Over Replacement Player) which is calculated using the specifications described above. (Learn more on VORP [here](https://en.wikipedia.org/wiki/Value_over_replacement_player)).
- **ADP** : A players ADP (Average Draft Position) calculated by averaging a players pick location from Sleeper and RTSports
- **RankvsADP** : Calculated by subtracting a players ADP from their Rank. If a player has a  RankvsADP>>0 they are *under-valued*, and if visa versa they are *over-valued*.



## Local Deployment

**Coming Soon**

### Using player data in repo
This project can be deployed using the pre-existing player data from [pro-football-reference](https://www.pro-football-reference.com/) and [fantasy-pros](https://www.fantasypros.com/) included in the **exports** directory.


## Appendix 

### Appendix A: Model Architecture

- Data Preprocessing: The first step involves loading the dataset and performing necessary data preprocessing steps. The unnecessary column is dropped, and categorical variables (e.g., "FantPos" and "Tm") are encoded using one-hot encoding. Numerical features are standardized to ensure each feature has a mean of 0 and a standard deviation of 1.

- Weighted Data: The dataset is sorted by player and year, and a decay factor is applied to calculate weights for each row. The decay factor allows recent data to have a higher impact on predictions. The PPR values and numeric features are multiplied by their respective weights to emphasize more recent data.

- Data Split: The weighted data is split into training and test sets. The training data includes records from 2010 to 2021, while the test data contains data from 2022.

- Gradient Boosting Regression: The model used for prediction is a Gradient Boosting Regressor, which is a powerful ensemble learning technique that builds multiple weak learners (decision trees) sequentially to improve prediction accuracy. The model is trained on the weighted training data.

- Prediction and Evaluation: The trained model is used to predict the PPR values for the year 2022. However, since there is no actual data for 2022, the predictions cannot be evaluated against the ground truth. The model's performance is assessed internally using Root Mean Squared Error (RMSE) for both PPR values and player ranks. RMSE measures the difference between predicted and actual values, providing insights into the model's accuracy.

- Deployment: This model is then applied to player data from 2010 to 2022 and retrained to predict PPR for 2023.