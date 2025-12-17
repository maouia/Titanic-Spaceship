# Spaceship Titanic – Kaggle Classification Project

This repository contains an end‑to‑end solution for the Kaggle “Spaceship Titanic” competition. The goal of the project is to predict whether each passenger was transported to an alternate dimension (`Transported`) based on demographic data, cabin information, and onboard spending features.

The project uses Python, pandas, and scikit‑learn, along with additional gradient boosting models (XGBoost and LightGBM), to build and compare several classification models. The workflow covers data loading, feature engineering, model training and evaluation, and generating a submission file in the correct Kaggle format.

## Dataset and Problem

The dataset comes from the Spaceship Titanic competition on Kaggle. Each row represents a passenger with features such as `HomePlanet`, `Destination`, `Cabin`, `Age`, `CryoSleep`, `VIP`, and spending in different services (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`). The task is a supervised binary classification problem: given these features, predict the target column `Transported` (True/False).

## Main Steps in the Notebook

- Exploratory data analysis (EDA) to understand feature distributions, missing values, and relationships with the target.  
- Data cleaning and preprocessing, including handling missing values and converting boolean/object columns into numeric form.  
- Custom feature engineering:
  - Total onboard spending (`TotalSpend`) and spending flags (e.g., `IsSpender`).  
  - Parsing `Cabin` into `Deck`, `CabinNum`, and `Side`.  
  - Group features derived from `PassengerId`, such as group ID, group size, and whether a passenger is traveling alone (`IsSolo`).  
  - Age imputation and age banding into meaningful age groups.  
  - Ensuring consistency between `CryoSleep` and spending columns.  
- Encoding categorical variables using one‑hot encoding for selected columns.  
- Building training and validation sets with `train_test_split`, using stratification on the target to preserve class balance.

## Models and Evaluation

The notebook trains and evaluates multiple models, including:

- `RandomForestClassifier`  
- `GradientBoostingClassifier`  
- `LogisticRegression`  
- `XGBClassifier` (XGBoost)  
- `LGBMClassifier` (LightGBM)

Models are evaluated using accuracy on a held‑out validation set. A simple model comparison loop logs the accuracy of each model, making it easy to see which approach performs best on the engineered feature set.

## Generating Kaggle Submission

The final part of the project applies the same feature engineering pipeline to the test dataset. The best‑performing model is then used to generate predictions for `Transported`. These predictions are combined with the original `PassengerId` column to create a submission file with the required structure:

- `PassengerId`  
- `Transported` (boolean)

The resulting DataFrame is saved as `submission.csv`, ready to be uploaded to Kaggle.
