# Share Bike Demand Prediction

## Table of Contents
* [Problem Statement](#problem-statement)
* [Step 1: Importing, Understanding the Data, and EDA](#step-1-importing-understanding-the-data-and-eda)
* [Step 2: Data Preparation for Building Model](#step-2-data-preparation-for-building-model)
* [Step 3: Model Building and Training](#step-3-model-building-and-training)
* [Step 4: Residual Analysis](#step-4-residual-analysis)
* [Step 5: Predictions (on test set) and Model Evaluation](#step-5-predictions-on-test-set-and-model-evaluation)
* [Answer to the Business Problem](#answer-to-the-business-problem)
* [Acknowledgements](#acknowledgements)

## Problem Statement
A share-bike company wants to understand the factors that impact the rental of their bikes, and predict the demand for their bikes under different conditions. Specially, they want to know:

- which features are significant in predicting the demand for their shared bikes?
- how the demand is correlated to the above factors?

For this purpose, a dataset of the share-bike rentals are provided containing the count of rental bikes for different values of features including temperature, humidity, wind speed, year, month, season, holidays, etc.

The objective is to analyse the data, develop a linear regression model, if applicable at all, and advise the business with feedback on the above business problems.

## Step 1: Importing, Understanding the Data, and EDA
The dataset was successfully imported, and inspected. The following were found:
- 730 rows in the dataset
- None of the columns have null values.
- Also, the type of the columns are as expected, based on the observation of the head() of the dataframe.
- dependent variable: 'cnt' or the target variable
- independent variable: the rest of the variables (features)
- No duplicated rows were observed.

### EDA
For EDA of the categorical columns, the unique values of the column were found and then were replaced with the expressions advised in the data dictionary file.
The most important categorical features that are contributing to the target variable, i.e. 'cnt' are shown in the figures below. It can be observed that moving from one category to another category within the column shows changes in the rental bike count (cnt).  
**'cnt' vs 'season'**

<img src="/images/season.png" width = 500>

**'cnt' vs 'yr'**

<img src="/images/yr.png" width = 500>

**'cnt' vs 'mnth'**

<img src="/images/mnth.png" width = 500>

**'cnt' vs 'weathersit'**

<img src="/images/weathersit.png" width = 500>

EDA of the numeric features is shown below in the pair-plot and the correlations in the heatmap.

<img src="/images/pairplots.png" width = 800>

<img src="/images/heatmap.png" width = 500>

Inspection of the pairplots and the heatmap of the correlations reveals that:

- There is strong positive linear correlation between (temp and cnt), and (atemp and cnt) and less strong negative correlation between (hum and cnt) and (windspeed and cnt).
- The absolute value of the correlation between (windspeed and cnt) is much higher than the correlation between (hum and cnt).
- Also, it can be seen that temp and atemp are highly linearly correlated. This means that either of these can be dropped, however, it is up to RFE to choose the most important features during model building.

**Need to consider a Multiple Linear Regression (MLR) model?**  
Investigating the categorical and numeric columns show that there is some linearity between some of the features (numeric and categorical) and the target (cnt). Therefore, a Multiple Linear Regression (MLR) model should be considered and implemented.

## Step 2: Data Preparation for Building Model
In this step the following have been implemented:

- dummy variables (columns) for the categorical columns are created and the original categorical columns were dropped.
- the dataframe has been split into train dataframe and test dataframes.
- the normal scaling (Min-Max scaling) is applied to the numeric columns of the train dataframe, since the rest of the columns are 0/1 's from dummy variables.
- dependent feature (y_train) and independant dataframe (x_train) has been created.

## Step 3: Model Building and Training
In this step the folowing have been implemented:

- RFE from sklearn is applied to choose the 20 most important features from the available features.
- Then, the 20 selected features are fine tuned (dropped) through stats model by closely monitoring the p-values and VIFs. This step is explained at each variable drop stage to mention the reason each feature has been dropped.

RFE has selected the following 20 feaures: 
'temp', 'atemp', 'hum', 'windspeed', 'spring', 'summer', 'winter', '2019', 'Dec', 'Feb', 'Jan', 'Jul', 'May', 'Nov', 'Sep', 'non-holiday', 'Sun', 'working-day', 'mist+clouds', 'snow'

Then, by a step-wise model building in stats model, and monitoring the p-value and VIF of the features, the following features have been dropped subsequently to drive the final MLR:
- Looking at the p-values and VIFs, it was seen that 'atemp' has a high p-value and a high VIF. Hence, 'atemp' was dropped first.
- Then, 'May' was dropped which had a high p-value and low VIF.
- Then 'Feb' was dropped with high p-value ad low VIF.
- Since p-value more than 5% (0.05) was considered as high, then 'non-holiday' was dropped with high p-value (of 0.054) and low VIF.
- Next, 'spring' was dropped with low p-value but high VIF (of more than 5).
- Then, 'Nov' was dropped with high p-value and low VIF.
- Next, 'Dec' was dropped with high p-value and low VIF.
- The next step would be to drop 'const' with low p-value and high VIF, however, statsmodel throws error and doesn't work without 'const'. 
Therefore, this model (with the following 13 features) will be considered as the MLR model for the problem:
'temp', 'hum', 'working-day', 'Sun', 'mist+clouds', 'Jan', 'winter', 'Jul', 'summer', 'snow', 'windspeed', 'Sep', '2019'

During the feature drop stage:

- the R squared decreased from 0.852 (for 20 features selected by RFE) to 0.845 (in the final MLR model).
- the Adjusted R squared increased from 0.846 (for 20 features selected by RFE) to 0.841 (in the final MLR model).

This means this final model captures 84.5% of the variance in the target variable by the available features.

All the available features have (details in the ipynb file):

- very low p-values, meaning they are all significant;
- and low VIFs, meaning they are all independent of each other.

## Step 4: Residual Analysis
The histogram and scatter plot of the residuals were produced as below.

<img src="/images/histogram.png" width = 500>

<img src="/images/scatter.png" width = 500>

Analysis of the histogram and the scatter plot of the residuals reveals that:

- the residuals follow a normal distribution with a mean of zero, as expected as one of the conditions of the MLR assumptions.
- the residuals are scattered around 0, and are independent of each other.
- the residuals have constant variance (homoscedasticity).

## Step 5: Predictions (on test set) and Model Evaluation
The test set values were predicted and a R Score of 0.805 was obtained as below:

<img src="/images/r2-score.png" width = 1000>

As can be seen, the R squared value of the test set is 80.5%, which is slightly different from the R squared value of the training set (84.5%), which is acceptable due to the insignificant difference between these two values. Therefore, it can be concluded that this model is able to generalise the results, and is not over-fitted to the train set.
  
  
## Answer to the Business Problem

To answer the qusetions raised by the business problem:
- the top 3 predictors for share bike rentals are: temperature (temp), snow conditions (snow) and the year 2019.
- the equation for interpreting the rental counts based on the above variables is:  
#### bike rentals = 0.5684 x temp -0.2425 x snow + 0.2296 x "2019" + the rest of the multipliers and features  
This means:  
- 1 unit increase in temperature (without increase in the rest of the features) increases the bike rentals by 0.56 units.  
- 1 unit increase in snow conditions, on the other hand, decreases the bike rentals by 0.24 units.

## Acknowledgements
- I would like to acknowledge the feedback, support and dataset provision by [upGrad](https://www.upgrad.com/gb) and The [International Institute of Information Technology (IIIT), Bangalore](https://www.iiitb.ac.in/).  
- Also, I would like to express my gratitude to [Nishan Ali](https://www.linkedin.com/in/nishan-ali-826552166/) for providing clarification and guidance to carry out this project.   
- The dataset of this paper was used and is acknowledged: Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3. 
- Furthermore, the valuable feedback from [Dr Tayeb Jamali](https://www.linkedin.com/in/tayeb-jamali-b1a10937/) is highly appreciated.
