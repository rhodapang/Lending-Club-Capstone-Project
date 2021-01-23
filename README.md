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

Generally Speaking, the columns that have not been located in User feature (general),User feature (financial specific),Secondary application info and Loan general feature would be excluded in the dataset.

# 2. Data Cleaning 

This step focus on collecting the data, organizing it and making sure it's well defined.



