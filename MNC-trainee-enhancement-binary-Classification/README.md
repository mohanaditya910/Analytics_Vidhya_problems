# Achieved top 10 percentile score in the Private Leader board of this competetion..

## Problem Statement:

A multi-national company is providing training to its employees on sales. So they want to decide whether the employee has passed the trainging given to him based on the data provided. Our job is to build a machhine learning model which can help the company in making better decisions.

The given problem is a **Binary classification problem** evaluated on the basis of **AUC_ROC_Score**. It contains both **Categorical and Continous** variables.

Mention the variable names.
**Train_data_count: ~73K**

## How I attempted to solve the problem..

### Preprocessing and Cleaning Data:

I found that some variables have missing values. They are 'age' and 'trainee_engagement_rating'. As the **'trainee_engagement_rating'** has very few missing (~70 out of 73k) values I directly substituted them with the mode of the variable. For **'age'** it was different altogether, there were around 30k missing values out of 73k instances of data. So, I cannot randomly fill the missing values of 'age' with mode or mean of the distribution as they can change the distibution of the 'age' entirely.

Then I mapped all the classes of categorical variables to numbers and then one-hot-encodded them so that numeric advantage is disabled for the classes (in a 3 class variable, if it is not one-hot-encodded then the class which is mapped to 2 has some unfair advantage).

All the Continuous variables are normalized.

I have dropped some variables which I thought were not useful in solving the problem and they are **'trainee_id','id','program_id','test_id'**. My thought was they are like roll numbers for a student and there is no relation between roll nnumber and student passing. That's why I dropped them.

Next step is filling the 'age'. I could not find a pattern of how to fill the 'age' with other variables. So, I thought of training a model for predicting the 'age' with the ~40k complete data(73k -missing 30k) and use this model to predict the missing values in 'age'. For this I have choosen **XGBRegeressor**.

As, I could not find any relations between variables which are dominated by categorical variables, I thought of using polynomial features with degree 2 and then use PCA to choose only relevant features.

### Model

I have choosen **XGBClassifier** with eval_metric='auc',objective='rank:pairwise', max_depth=10. These parameters were giving the best score (0.799) after getting in them in gridsearch.

rank:pairwise: Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized.

## What the best approches were??

I dropped 'test_id' and 'trainee_id' without checking their importance. By looking at the name, I thought that they wont be important. That is a major setback as they are the important features in other's models. So, I learnt that don't drop any feature without checking its importance and let some statistics be there to support you in dropping that variable.

Mostly, they were using XGBOOST, lightgbm a lighter version.










