# Predicting Default Rate for Lending Club Dataset

[LendingClub](https://www.lendingclub.com/) is a US peer-to-peer lending company and the world's largest peer-to-peer lending platform.  Since these loans are unsecured and companies creating the market generally do not invest their own capital, neither borrowers nor companies assume any risk. Entire credit risk is born by investors. Therefore, the purpose of this data science project is to come up with a delinquency prediction model for the Lending Club. These models could help LendingClub investors make better-informed investment decisions.

In this project, I build machine learning models to predict the probability that a loan on LendingClub will charge off (default).  I use a 1.8 GB LendingClub dataset with 1,646,801 loans and 150 variables for each loan.

# 1. Data 

The dataset is from [Lending Club platfrom](https://www.lendingclub.com/investing/peer-to-peer). I download the dateset on 2018Q1 with 107,864 loans and 100 variables for each loan.

From the document-LCDataDictionary provided by lending club, there are total categories variables in the dataset:
- User feature (general)
- User feature (financial specific): income, credit scores, credit lines
- Loan general feature
- Loan payment feature
- Current loan payment feature
- Secondary application info
- Hardship
- Settlement
- Potential response variables:sub_grade, int_rate, loan_status

# 2. Data Cleaning in [Lending Club Data Wrangling-1](https://github.com/rhodapang/Lending-Club-Capstone-Project/blob/main/Lending%20Club%20Data%20Wrangling-1.ipynb)
This step focus on collecting the data, organizing it and making sure it's well defined.

1. Generally Speaking, the columns that have not been located in User feature (general),User feature (financial specific),Secondary application info and Loan general feature would be excluded in the dataset. The response variabe in the model is loan_status, the features categories should be included in the model are User feature (general),User feature (financial specific), Secondary application info

2.The column of ID, url, issue_d are irrelevant to the model building, so they could be dropped, too.
- ID:A unique LC assigned ID for the loan listing.
- issue_d:The month which the loan was funded
- url: URL for the LC page with listing data.

3. Before understand the deep meaning of all columns, the columns with all missing value could be dropped firstly.

4. Drop columns with only containing the distinct value.

5. Drop the deplicated rows in dataframe

6. Check the correlations between these variables. Explore weather there are some columns which are highly multicolinearity,if yes, it could be dropped one of them due to the lots of columns existing.High Correlation Coefficients Pairwise correlations among independent variables might be high (in absolute value). Rule of thumb: If the correlation > 0.8 then severe multicollinearity may be present.

# 3. EDA in [Lending Club Exploratory Data Analysis-2](https://github.com/rhodapang/Lending-Club-Capstone-Project/blob/main/Lending%20Club%20Exploratory%20Data%20Analysis%20-%202.ipynb)

## 1. Univariate Analysis
- Numerical Variables
- Categorical Variable
## 2.Bivariate analysis
- Numerical Variables
- Categorical Variable

# 4. [Preprocessing the Lending Club Dataset](https://github.com/rhodapang/Lending-Club-Capstone-Project/blob/main/Lending%20Club%20Preprocess%20the%20dataset-3.ipynb)
- Deal with the missing values
- Numerical variables transformation: conducting log transformation for 'dti' and 'annual_inc' and standardize the continuous variables for the next machine learning model applied
- Categorical variables transformation: get dummies for the categorical variables 

# 5. [Machine learning Model Apply for predicting default rate the loans from Lending Club](https://github.com/rhodapang/Lending-Club-Capstone-Project/blob/main/Lending%20Club%20Machine%20Learning%20Models%20Applied%20-%204.ipynb)
## 1. Apply for machine learning models 
Applying the models for the preprocessed dataset and there are three models including Logistic Regression Model, Random Forest Model and Xgboosting Model. 

## 2. Model Comparison 
Below are the deatil result of each model performance and all of them achieve great score. 
              train_auc	          test_auc
logit_model	  0.9492	             0.9487
rf_model	    0.9432	             0.9265
xgb_model    	0.9588	             0.9513

## 3. Model Validation 
- Validation the model among continous varibles
The variable of 'last_fico_range_high' show high correlation with defaul rate and even reach to 0.8 and others variables mainly surround 0.02~ 0.2.
therefore, it should be pay attention when on the feature selection stage. 

- Validation the model among categorical varibles
There is no any abnormal varibles after validation 

## 4. Feature Selection 
In the feature selection, the plot show the importance of 'last_fico_range_high' reaches above 0.7, which means that we should take a deep research for this variable again. 

Definition of last_fico_range_high: The upper boundary range the borrower’s last FICO pulled belongs to.

When a borrower applies for a loan, LendingClub gets the borrower’s credit score from FICO — they are given a lower and upper limit of the range that the borrowers score belongs to, and they store those values as fico_range_low, fico_range_high. After that, any updates to the borrowers score are recorded as last_fico_range_low, and last_fico_range_high, which means that this variables is related to target variable:loan_status. 

The solution is that this variable should be dropped and run the model again. 

## 5. Fit the models and validate the models again
Below is the performance data for each model 

              t rain_auc	test_auc
logit_model	  0.7012	    0.7062
rf_model	    0.7431	    0.7024
xgb_model	    0.7200	    0.7116

Below is the classification report for each model 
logit_model
              precision    recall  f1-score   support

           0       0.86      1.00      0.92     18536
           1       0.54      0.01      0.02      3037

    accuracy                           0.86     21573
   macro avg       0.70      0.50      0.47     21573
weighted avg       0.81      0.86      0.80     21573

rf_model
              precision    recall  f1-score   support

           0       0.86      1.00      0.92     18536
           1       0.00      0.00      0.00      3037

    accuracy                           0.86     21573
   macro avg       0.43      0.50      0.46     21573
weighted avg       0.74      0.86      0.79     21573


xgb_model
              precision    recall  f1-score   support

           0       0.86      1.00      0.92     18536
           1       0.75      0.00      0.01      3037

    accuracy                           0.86     21573
   macro avg       0.80      0.50      0.47     21573
weighted avg       0.84      0.86      0.80     21573

# Conclusion 
- From the data providing for classification report, we decide to adopt Xgboosting model as the final model. 
- The mode feature selection, we know that the top 10 important features including int_rate, grade_A, home_ownership_mortgage, mac_bal_bc, loan_amnt, mths_since_reacent_inq, revol_util, verification_status_Not_Verified, open_rv_24m, tot_cur_bal. 







