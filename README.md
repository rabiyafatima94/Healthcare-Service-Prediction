# Patient Characteristics to Predict the Type of Healthcare Service 
Explore the future of healthcare with our machine learning models! Predict patient needs, optimize resource allocation, and foster proactive interventions. Join us in reshaping healthcare analytics.

Predictive Modeling for Healthcare Service Utilization

## Overview

This repository contains the code, docs and presentation materials for a predictive modeling project focused on healthcare service utilization. The project explores the use of machine learning techniques to forecast the type of healthcare service a patient might require based on diverse patient attributes.

## Project Structure

- **Code**: The Python code for data preprocessing, feature engineering, model training, and evaluation is available in the `code` directory.
  
- **Presentation**: The PowerPoint presentation summarizing the project, including methodology, findings, and recommendations, can be found in the `presentation` directory.

## Dataset

The project utilizes a diverse dataset, including original patient data and integrated information on demographic and environmental factors. The data integration process involved aligning datasets based on a common attribute, namely Zip code, to create a more comprehensive dataset for analysis.

https://catalog.data.gov/dataset/patient-characteristics-survey-pcs-2019

Patient Characteristics Survey (PCS): 2019
Metadata Updated: November 29, 2021

The data are organized by OMH Region‐specific (Region of Provider), program type, and by the following demographic characteristics of the clients served during the week of the survey: 
sex (Male, Female, and Unknown), Transgender (No, Not Transgender; Yes, Transgender and Unknown), age (below 17 (Child), 18 and above(Adult) and unknown age) and 
race (White only, Black Only, Multi‐racial, Other and Unknown race) and ethnicity (Non‐Hispanic, Hispanic, Client Did Not Answer and Unknown). 


## Key Steps

1. **Data Pre-Processing:**
    - Integration of alternative datasets using Zip codes.
    - Removal of irrelevant columns and handling null values.
    - Standardization and normalization of data.

2. **Feature Engineering:**
    - Conversion of nominal and ordinal values to numerical formats.
    - Correlation analysis for feature selection.
    - Principal Component Analysis (PCA) for dimensionality reduction.

3. **Data Mining Models:**
    - Training and evaluating various machine learning models, including Random Forest, Decision Tree, Gradient Boosting, Neural Networks, Logistic Regression, and Naive Bayesian.
    - Evaluation metrics include precision, recall, accuracy, and F1 score.

4. **Results and Recommendations:**
    - Identification of top-performing models for predicting healthcare service utilization.
    - Recommendations for healthcare resource allocation and proactive interventions.

## Limitations and Future Work

The project acknowledges limitations, such as challenges in predicting emergency services and the reliance on 2019 data. Future work includes updating the dataset, external validation, and collaboration with healthcare institutions for region-specific data.

## How to Use

1. **Code Execution:**
    - Open the `code` directory and follow the instructions in the Python files for data preprocessing, model training, and evaluation.

2. **Presentation Review:**
    - Open the `presentation` directory to access the PowerPoint presentation for an overview of the project.

Feel free to explore the code and presentation materials to gain insights into the project's methodology, findings, and recommendations. If you have any questions or feedback, please reach out.


## Contributors

- Rabiya Fatima
- Srilakshmi Mallipudi
- Gautam Reddy
- Barkha Sharma

  

