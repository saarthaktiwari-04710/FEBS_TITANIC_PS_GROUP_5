# FEBS_TITANIC_PS_GROUP_5

This task aims to predict whether a passenger was Transported (True/False) based on different features such as age, spending habits, cabin details, and travel information.

We both then first divided the task of handling with the missing values, and created two python files in our own branches with the code handling the missing values.

**Saarthak Tiwari:-**

1. `train.py`
   1. Handled missing values in columns:
      - Cabin
      - CryoSleep
      - Name
      - RoomSrvice
      - FoodCourt
      - Spa (and trained the SimpleImputer using `fit_transform`).
   2. Plotted graphs of:
      - Transported Count
      - Age Distribution
      - Boxplot of transported vs total spending
      - Graph between cryosleep and transported
   3. Wrote the code for Logistic Regression (Sigmoid function, iterating and using Gradient Descent, predict function).
   4. Made a dictionary artifacts and loaded it to the binary file `model_artifacts.pkl`.

2. `test.py`
   1. Used the trained SimpleImputer to fill the missing values in columns:
      - Cabin
      - CryoSleep
      - Name
      - RoomSrvice
      - FoodCourt
      - Spa
   2. Loaded values of w and b from `model_artifacts.pkl` using pickle.
   3. Made the prediction column using predict function.
   4. Created DataFrame submission and the file `submission.csv` for the final result.

