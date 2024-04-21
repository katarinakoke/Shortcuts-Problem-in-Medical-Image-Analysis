import pandas as pd
from sklearn import metrics
from sklearn.metrics import average_precision_score
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio

def area_under_the_curve(y_pred, y_true):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def average_precision(y_pred, y_true):
    ap_score = average_precision_score(y_true, y_pred)
    return ap_score

def main():
	# A list to store the results
	results = []
	count = 0

    # Opening the predictions folder
	for filename in os.listdir('.'):

        # Taking the prediction files only
		if filename.endswith('.csv'):
			if count >= 2:
				break

			# Getting the model name
			model_name = filename.rstrip('.csv')

			# Reading the csv file
			file = pd.read_csv(filename)

			y_pred = file['y_pred_Problem_in']
			y_true = file['Pneumothorax'].map({1.: 1 , 0.: 0, -1.: 0}).fillna(0)
			sensitive_features = file['Sex']
			
            # Area under the curve
			auc = area_under_the_curve(y_pred, y_true)
			
            # Average precision
			avg_precision = average_precision(y_pred, y_true)
			
            # Fairness
			eo_difference = equalized_odds_difference(y_true, y_pred, sensitive_features)
			eo_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features)
			
            # Append results for this model to the results list
			results.append({
                'Model Name': model_name,
                'AUC': auc,
                'Average Precision': avg_precision,
                'Equalized Odds Difference': eo_difference,
                'Equalized Odds Ratio': eo_ratio
            })
			count += 1
			
    # Saving the results
	results_df = pd.DataFrame(results)
	results_df.to_csv('model_evaluation_results.csv', index=False)

if __name__ == '__main__':
	main()