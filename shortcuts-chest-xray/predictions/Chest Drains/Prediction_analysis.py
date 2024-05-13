import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio
from sklearn.metrics import accuracy_score

def area_under_the_curve(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def average_precision(y_true, y_pred):
    ap_score = average_precision_score(y_true, y_pred)
    return ap_score

def main():
	# A list to store the results
	results = []

    # Opening the predictions folder
	for filename in os.listdir('.'):

        # Taking the prediction files only
		if filename.endswith('.csv'):

			# Getting the model name
			model_name = filename.rstrip('.csv')
			print(model_name)

			# Reading the csv file
			file = pd.read_csv(filename)

			y_pred = file['y_pred_Problem_in']
			y_true = file['Pneumothorax'].map({1.: 1 , 0.: 0, -1.: 0}).fillna(0)
			sens_features = file['Drain']

			sens_y_pred = file['sensitive_y_pred_Problem_in']
			sens_y_true = file['Drain']
			# sens_y_true = file['Drain'].map({'Female': 1 , 'Male': 0, 'Unknown': 0}).fillna(0)

			# Accuracy
			acc = accuracy_score(y_true, y_pred)
			sens_acc = accuracy_score(sens_y_true, sens_y_pred)
			
            # Area under the curve
			auc = area_under_the_curve(y_true, y_pred)
			sens_auc = area_under_the_curve(sens_y_true, sens_y_pred)
			
            # Average precision
			avg_precision = average_precision(y_true, y_pred)

            # Fairness
			eo_difference = equalized_odds_difference(y_true, y_pred, sensitive_features=sens_features)
			eo_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features=sens_features)
			
            # Append results for this model to the results list
			results.append({
                'Model Name': model_name,
				'Accuracy': acc,
				'Drain Accuracy': sens_acc,
                'AUC': auc,
				'Drain AUC': sens_auc,
                'Average Precision': avg_precision,
                'Equalized Odds Difference': eo_difference,
                'Equalized Odds Ratio': eo_ratio
            })
			
    # Saving the results
	results_df = pd.DataFrame(results).sort_values(by='Model Name')

	# Datasets
	results_df['TrainDataset'] = results_df['Model Name'].str.split('_').str[1]
	results_df['TestDataset'] = results_df['Model Name'].str.split('_').str[2]

	# Seed
	results_df['Seed'] = results_df['Model Name'].str.split('_').str[-2]

	# Lambda values
	results_df['Lambda'] = results_df['Model Name'].str.split('_').str[-1]
	lambda_mapping = {
    'neg01': -0.1,
    'neg005': -0.05,
    '0': 0,
    'pos005': 0.05,
    'pos01': 0.1}

	results_df['Lambda'] = results_df['Lambda'].map(lambda_mapping)

	# Saving the csv file
	results_df.to_csv('Prediction_analysis_chest_drains.csv', index=False)

if __name__ == '__main__':
	main()