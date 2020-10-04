from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import datetime
import numpy as np
import seaborn as sns
import pandas as pd

## Load the data

def read_data(url):
	df = pd.read_csv(url)

	return df 

## Explore the dataframe

def explore_distributions(df):
	print('Exploring distributions of numeric variables')
	sns.set(rc={'figure.figsize':(20, 15)})
	numericvars = list(df.select_dtypes(include=(int, float)))
	df[numericvars].hist()
	
def explore_correlations(df):
	print('Exploring correlations')
	df.corr()
	plt.matshow(df.corr())
	
def lookfor_outliers(df):	
	print('Looking for outliers')
	df.describe().round()

def identifyng_timerange(df):	
	print('Identifying time range of the data')
	datevar = list(df.select_dtypes(include=np.datetime64))
	startdate = df[datevar].min()
	endate = df[datevar].max()
	print('Starting date: {} - Ending date: {}'.format(startdate[0], endate[0]))

## Create Training and Testing Sets

def split_dataset(df, features, target):
	x = df[features]
	y = df[target]
	X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=42)

	return X_train, X_test, y_train, y_test

## Pre-Process the Data

def convert_numerical(y_train, y_test, namecolumn):
	y_train[namecolumn] = y_train[namecolumn].astype('int64')
	y_test[namecolumn] = y_test[namecolumn].astype('int64')

	return y_train, y_test

def impute_missing(df, df_train, listcolumns):
	df_copy = df.copy()
	for column in listcolumns:
		imp = SimpleImputer(missing_values=np.nan, strategy='median')
		imp.fit(df_train[column].values.reshape(-1, 1))
		df_copy[column] = imp.fit_transform(df[column].values.reshape(-1, 1))
	
	return df_copy
		

def normalize_continuous(df, df_train, listcolumns):
	df_copy = df.copy()
	for column in listcolumns:
		df_copy[column] = ((df.loc[:,(column)] - df_train.loc[:,(column)].mean()) / df_train.loc[:,(column)].std())

	return df_copy

## Generate features
def generate_hotenconding(df_test, df_train, listcolumns):
	df_train_processed = pd.get_dummies(df_train, prefix_sep ="__", columns = listcolumns)
	df_train_dummies = [col for col in df_train_processed if "__" in col and col.split("__")[0] in listcolumns]
	processed_columns = list(df_train_processed.columns[:])
	df_test_processed = pd.get_dummies(df_test, prefix_sep ="__", columns = listcolumns)
	#Remove extra columns 
	for col in df_test_processed.columns:
		if ("__" in col) and (col.split("__")[0] in listcolumns) and col not in df_train_dummies:
			df_test_processed.drop(col, axis=1, inplace=True)

	for col in df_train_dummies:
		if col not in df_test_processed.columns:
			df_test_processed[col] = 0

	df_test_processed = df_test_processed[processed_columns]

	return df_train_processed, df_test_processed

def discretize_continuous(df, df_test, df_train, bins, listcolumns):
	for column in listcolumns:
		interval_range == pd.interval_range(start=listcolumns[column].min(), freq=listcolumns[columns].max() / bins , 
			end=listcolumns[column].max(), closed='left')
		df_test[column + "_cut"] = pd.cut(df_test[column], bins=interval_range)
		df_train[column + "_cut"] = pd.cut(df_train[column], bins=interval_range)

	return df_test, df_train

##Evaluate classifiers 

def evaluate_classifiers(y_test, y_predicted):
	classifiers_dict = {}
	classifiers_dict['Accuracy score'] = (accuracy_score(y_test, y_predicted))
	classifiers_dict['F1 Score'] = (f1_score(y_test, y_predicted))
	
	return classifiers_dict

## Build classifiers 

def build_classifier(X_train, X_test, y_train, y_test, MODELS, GRID ):
	start = datetime.datetime.now()

	result_df = pd.DataFrame({"Training Model": [], "Parameters": [], "Metrics":[]})
	# YOUR CODE HERE

	# Loop over models 
	for model_key in MODELS.keys(): 

		# Loop over parameters 
		for params in GRID[model_key]: 
			print("Training model:", model_key, "|", params)

			# Create model 
			model = MODELS[model_key]
			model.set_params(**params)

			# Fit model on training set 
			model.fit(X_train, y_train)
			
			# Predict on testing set 
			y_predicted = model.predict(X_test)
			
			# Evaluate predictions
			evaluation = evaluate_classifiers(y_test, y_predicted)

			# Store results in your results data frame 
			# YOUR CODE HERE
			result_df = result_df.append({"Training Model": model_key, "Parameters": params, "Metrics": evaluation}, 
				ignore_index = True)

	# End timer
	stop = datetime.datetime.now()
	print("Time Elapsed:", stop - start)

	return result_df


