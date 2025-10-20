Car Recommendation Pipeline
Overview
This project implements a car recommendation system using clustering and classification techniques. It processes raw car data, performs feature engineering and clustering, and finally provides top N car recommendations based on customer inputs.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Dataset
Raw Data: Cars-Datasets-2025-2.csv
Contains all car information used for analysis and recommendation.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Project Files
1.cars_datasets.py
Performs initial data cleaning and preprocessing on the raw dataset.

2.Clustering.py
Uses the preprocessed data (df_final) to perform feature engineering and clustering.
The final selected clusters are saved to clustered_cars_data.csv.
clustered_cars_data.csv contains the processed and clustered car data, which serves as input to the classification model.

3.Classification_Pipeline.py
Uses the clustered data to provide top N car recommendations based on customer inputs.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Usage Instructions

Step 1: Install Required Packages
Ensure all required Python packages are installed (e.g., pandas, scikit-learn).
Use the provided requirements.txt to install dependencies:

pip install -r requirements.txt


Step 2: Preprocess Raw Data
Run cars_datasets.py to preprocess the raw dataset.

Input: Cars-Datasets-2025-2.csv (raw dataset)

Output: df_final (preprocessed dataframe)

Step 3: Perform Clustering
Run Clustering.py to perform feature engineering and clustering using the preprocessed dataframe.

Input: df_final (from Step 2)

Output: clustered_cars_data.csv (clustered dataset ready for classification)

Step 4: Generate Recommendations
Run Classification_Pipeline.py to generate car recommendations based on customer input.

Input: clustered_cars_data.csv (from Step 3)

Output: top_recommendations.csv (top 5 car recommendations)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Final Outputs

clustered_cars_data.csv – Clustered and processed car data


top_recommendations.csv – Top N car recommendations based on customer input
