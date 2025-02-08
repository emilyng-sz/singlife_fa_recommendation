# Customer-Advisor Recommendation System

Authors: Geoff Ong, Emily Ng Siew Zhang, Khine Ezali, Louise Tan

This project is our submission for 6-day Datathon held by NUS Statistics and Data Science Society (SDSS) and Singlife Singapore from 1st to 6th February 2025.

## Table of Contents
- **[About the Project](#about-the-project)**<br>
- **[Key Insights](#key-insights)**<br>
- **[How to run the project](#how-to-run-the-project)**<br>
- **[Using our Model ](#using-our-model)**<br>


## About the Project
To enhance client-advisor collaborations in the financial services industry, we propose a solution that leverages machine learning to match the most suitable financial advisors for customers. Our approach involves training multiple Random Forest Classifier models to learn patterns from past agent-client interactions. This project provides a novel and efficient method for more personalised and efficient client-advisor collaborations.

## Key Insights
With multiple Random Forest Classifiers to predict individual agent features, along with a cosine similarity metric, we achieved a good performance with a **33% accuracy**, where accuracy is defined as the correct agent being in the top 10 preferred agents based on our model. 

Considering that there are 8767 unique agents, we believe that being able to predict a correct match for 33% of the clients is already a significant achievement since our model is able to predict the correct agent within 1% of all available agents.

  
## How to run the project
1. Clone this repository to your local machine
2. Create a `data/` folder where all data will be stored
- Add the raw data files `nus_agent_info_df.parquet`, `nus_client_info_df.parquet` and `nus_policy_info_df.parquet` to the folder. Note that these files are not included in the repository due to data privacy reasons
3. Create a `model/` folder where the fitted models will be exported to
4. Ensure that Python version 3.12 is installed and in use
5. Navigate to the project directory and install the required dependencies:
```bash
pip install -r requirements.txt
```
6. Run the `CAT_A_3_final.ipynb` file to reproduce our project. 
- Note that running the notebook will by default also export the cleaned data to `data/` and fitted models to `model/` in `.pkl` format.  
- To change these settings, modify the `EXPORT_DATA` and `EXPORT_MODELS` variables at the start of the notebook

## Using our Model 

### Preparing Input Data 
Based on the data cleaning, merging and pre-processing in `CAT_A_3_final.ipynb`, ensure the new input has the columns in the same order as `X_variables` below. Furthermore, ensure the agent data is also preprocessed accordingly with columns in the same order as `y_labels` below.

```
X_variables = ['cust_age_at_purchase_grp_AG01_lt20',
'cust_age_at_purchase_grp_AG07_45to49',
'cust_age_at_purchase_grp_AG08_50to54',
'cust_age_at_purchase_grp_AG09_55to59',
'cust_age_at_purchase_grp_AG10_60up','economic_status', 'age',
'cltsex_M', 'marryd_M', 'marryd_P', 'marryd_S', 'marryd_W',
'race_desc_map_Indian', 'race_desc_map_Malay', 'race_desc_map_Others',
'household_size_grp_HH2_40to80', 'household_size_grp_HH3_80to100',
'household_size_grp_HH4_100to120', 'household_size_grp_HH5_120up',
'family_size_grp_FS2_20to40', 'family_size_grp_FS3_40to60',
'family_size_grp_FS4_60to80', 'family_size_grp_FS5_80up']

y_labels = ['agent_age', 'agent_gender_M', 'agent_marital', 'agent_tenure', 'cnt_converted_cat', 'annual_premium_cnvrt_cat', 'agntnum']
```
Things to Note:
`cust_age_at_purchase_grp_X` is the only variable taken from the Policy Information table. All other variables are taken from the Client Information table. If this info is not available at the time of prediction, use the age (of the client) to fill this row

### Using our pre-trained models

For each `label_name` in `y_labels`, load the fitted model in pickle format using:
```
with open(f"{label_name}_model.pkl", "rb") as file:
               model = pickle.load(file)
```
Each model should be used to predict the values for the respective `label_name`. Then, combine all predicted columns in the same order and run the cosine similarity metric comparison to get the top 10 most likely agents to fit the client’s preference. 