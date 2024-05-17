# Supplementary Information - Source Code

Source code of article 'Roboticized AI-assisted microfluidic photocatalytic synthesis and screening up to 10,000 reactions per day'.

This repository provides the data set and the source code related to the regression models.

## 1. Python environment configuration
The code was written and tested in **Python 3.6.13**.

Python packages that need to be installed: **pandas(1.1.5), numpy(1.19.2), scikit-learn(0.24.2), xgboost(1.5.2), deep-forest(0.1.5), PyQt5(5.15.4)**.

Users can also try using other versions of Python and packages.

## 2. User guide
**2.1. User interface**

After completing the above configuration, users can run the **RegressionModels\_UI.py** file, If all configurations are OK, the program will display the following user interface.

![image](https://github.com/WangJianwei1991/LJM_Regression/assets/35262865/cfd24674-9355-4550-967f-a72881468453)

**2.2. Usage**

1. Download the dataset.xlsx file. 
2. Run the **RegressionModels\_UI.py** file.
3. Click the '**Load Dataset**' button to load the data set.
4. The **5** in ‘**Which Excel worksheet is the dataset in? (Index start from 0)**’ edit box means the data we want to use is in the 6th worksheet of the xlsx file. The index value starts from 0, and the index value of the 6th worksheet is 5. Here, the user does not need to modify.
5. The **8** in '**Non-steady state data column index start value**' edit box means the index value of the starting column of non-stationary data is 8. Here, the user does not need to modify.
6. The **40** in '**Non-steady state data column quantity**' edit box means the number of columns of Non-steady state data is 40 columns. Here, the user does not need to modify.
7. Select PCA Mode. A total of 3 PCA processing methods are built-in, namely 'Mode 1(PCA Non-steady state data)', 'Mode 2(PCA all data)', 'Mode 3(no PCA)'.
8. Select Train set ratio. There are 2 built-in options, namely **[0.7]**, **[0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]**. Users can modify, add or delete values ​​within options. But don't destroy the structure of the list, and make sure there is at least one value in the list. For example, we can change **[0.7]** to **[0.6]**, but do not delete the **[]** on both ends of 0.6
9. Select a model. A total of 12 regression models are built-in.
10. **random\_state** is a parameter in train\_test\_split that controls the random number generator used to shuffle the data before splitting it. Here we set it to 42.
11. **GridSearchCV - cv** means the fold of cross validation used for grid search. Here we set it to 5, users can also set it to 10 or other values.
12. **GridSearchCV - Parameters Grid** means hyperparameter information of the model used for grid search. Here, users can modify the hyperparameter name and value range, but you must make sure that the hyperparameter name and value range you modify are valid.
13. Select the folder to save the model prediction results file for the test set.
14. Click the '**Train, Predict and Report**' button to start data set partitioning, data set preprocessing, grid search (model training, cross-validation evaluation and hyperparameter optimization), and use the trained model to predict the test set, and calculate the performance indicators of the model on the test set, And save the result file to the specified folder.
