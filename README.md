# Fashion Retail Intelligence - Predictive Modeling
This project focuses on building a predictive model to forecast product demand using historical sales and product information.

We experimented with different classification algorithms, tackled class imbalance, tuned hyperparameters, and analyzed feature importance to extract business insights.

The goal was to develop a robust model that can identify products with high purchase likelihood, enabling better inventory and marketing decisions.

# Project Workflow
## Data Understanding & Preprocessing
Loaded dataset containing product details, prices, and demand indicators.

Handled missing values (imputation where necessary).

Encoded categorical variables using appropriate encoding techniques.

Scaled numerical features where required.

# Baseline Model – Decision Tree Classifier
Built a Decision Tree to establish a baseline performance.

Observed high accuracy for the majority class but low recall for the minority class.

Identified class imbalance as a key challenge.

# Handling Class Imbalance
Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the training dataset.

Improved recall for minority class predictions.

# Hyperparameter Tuning
Used GridSearchCV for Decision Tree:

Tuned max_depth, min_samples_split, criterion, and min_samples_leaf.

Focused on F1-score to balance precision and recall.

# Random Forest Classifier
Built a Random Forest model to test ensemble performance.

Applied SMOTE + GridSearchCV for hyperparameter tuning.

Results:
* Slight improvement in accuracy.
* Minority class performance still limited — suggesting lack of strong predictive features.

# Feature Importance Analysis
Extracted feature importances from the tuned Random Forest model.

Identified the most influential predictors for demand.

# Model Performance Summary
Model	            Imbalance Handling	    Hyperparameter Tuning	     Accuracy	    Minority Class F1
Decision Tree	          No	                    No	                   High	        Very Low
Decision Tree	         SMOTE	                  No	             Improved Recall	  Moderate
Decision Tree	         SMOTE	                 Yes (GridSearchCV)	   Balanced	     Improved F1
Random Forest	         SMOTE	                 Yes (GridSearchCV)	     High         	Low

# Key Learnings
Real-world datasets often suffer from class imbalance, which can mislead accuracy metrics.

SMOTE can improve minority class detection but may not always lead to major performance boosts without strong predictors.

Hyperparameter tuning can significantly optimize model behavior.

Even when prediction scores aren’t perfect, feature importance analysis can deliver valuable business insights.

# Tech Stack
Python

Pandas, NumPy – Data processing

scikit-learn – Modeling, SMOTE, GridSearchCV

Matplotlib, Seaborn – Visualization
