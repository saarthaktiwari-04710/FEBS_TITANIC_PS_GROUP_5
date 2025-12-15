# FEBS_TITANIC_PS_GROUP_5

This task aims to predict whether a passenger was Transported (True/False) based on different features such as age, spending habits, cabin details, and travel information.

We both then first divided the task of handling with the missing values, and created two python files in our own branches with the code handling the missing values.

**Saarthak Tiwari:-**

I. `train.py`
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

II. `test.py`
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

        
**Thanvi Chennupati:-**

I. `train.py`
   1.Handled missing values in columns:
     -Homeplanet
     -Destination
     -VIP
     -Shopping Mall
     -Age
     -VRDeck
     
   For spending-related columns, missing values were replaced with 0 because a missing entry likely indicates no spending.
   Missing Age values were filled using the median.
   Missing values for VIP filled false.
   Missing values in HomePlanet and Destination were filled using the mode since these are categorical variables.

   Created a variable for Total Spending and sum up all the spendings into that , to make the analysis easy.

  2. Plotted graphs of:
     -Homeplanet vs transported
     -Destination vs transported
     -Age vs transported
     -Side vs transported
     -Deck vs transported

  3.Dropped the columns that are not useful for the training of the model.
  
  4.Did the encoding for Homeplanet, Destination , side , deck , VIP and cryosleep using one hot encoder.

  5.Changed the values of Transported from T/F to 1/0.

  6.Did target splitting for the model , x as features and y as transported or not.

  7.The numerical features Age and TotalSpending were scaled using StandardScaler.

II. `test.py`

   1.Again used the old file to fill missing values in columns:
     -Homeplanet
     -Destination
     -VIP
     -Shopping Mall
     -Age
     -VRDeck
     
   2.Repeated all the steps like dropping few unimportant columns and creating new ones for total spending and did encoding for above mentioned columns again using one-hot encoder

   
   Uploaded all the required data into the repo (train.csv,test.csv,submission.csv) and checked all the code for errors and corrected if any.
   

