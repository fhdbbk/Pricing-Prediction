import numpy as np
import pandas as pd
from utils import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model

STATES = {"Andhra Pradesh" : 0, "Arunachal Pradesh" : 1, "Assam" : 2, "Bihar" : 3, "Chhattisgarh" : 4,
		  "Goa": 5, "Gujarat" : 6,"Haryana" : 7,"Himachal Pradesh" : 8, "Jammu and Kashmir" : 9, "Jharkhand" : 10, 
		  "Karnataka" : 11, "Kerala" : 12, "Madhya Pradesh" : 13, "Maharashtra" : 14, "Manipur" : 15,"Meghalaya" : 16,
		  "Mizoram" : 17, "Nagaland" : 18, "Odisha" : 19, "Punjab" : 20, "Rajasthan" : 21, "Sikkim" : 22,"Tamil Nadu" : 23,
		  "Telangana" : 24,"Tripura" : 25, "Uttar Pradesh" : 26, "Uttarakhand" : 27, "West Bengal" : 28}

EPC_VENDORS = {"LNT" : 0, "Nagarjuna" : 1, "GMR" : 2, "Tata" : 3, "Gammon" : 4}


def classify(X, y):
	with open('classifier_parameters.pickle', 'rb') as handle:
		saved_parameters = pickle.load(handle)
	pred_test = predict(X, y, saved_parameters)
	result = int(np.squeeze(pred_test))
	print("Value of result: {}".format(result))
	return result

def regress(X_res):
	data = pd.read_csv('data3.csv')
	data['Geography'] = data['Geography'].map(STATES)
	data['EPC Vendors'] = data['EPC Vendors'].map(EPC_VENDORS)
	X = data.drop(columns=['Historical Quote','Bid Success'],axis=0)
	X = X.values
	mean_X = np.mean(X)
	std_X = np.std(X)
	X = (X - mean_X)/std_X
	y = data['Historical Quote']
	y = y.values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
	lm = linear_model.LinearRegression()
	lm.fit(X_train, y_train)
	X_res = (X_res - mean_X)/std_X
	pred = lm.predict(X_res)[0]
	print("Predicted value of Quote: {}".format(pred))
	return pred

if __name__ == '__main__':
	data = pd.read_csv("data2.csv")
	data['Geography'] = data['Geography'].map(STATES)
	data['EPC Vendors'] = data['EPC Vendors'].map(EPC_VENDORS)
	with open('max_values.pickle', 'rb') as handle:
		max_values = pickle.load(handle)
	max_vals = max_values['max_values']
	
	X = data.values[:, 1:]
	X = X / max_vals
	Y = data.values[:, 0]
	Y = Y.reshape(len(Y), 1)

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

	X_train = X_train.T
	X_test = X_test.T
	y_train = y_train.T
	y_test = y_test.T

	with open('classifier_parameters.pickle', 'rb') as handle:
		parameters = pickle.load(handle)
	pred_train = predict(X_train, y_train, parameters)
	pred_test = predict(X_test, y_test, parameters)

	out = np.array([3, 24, 25993, 59]).reshape(1, 4)
	print(out.shape)
	out = out / max_vals
	print(out.shape)
	print(max_vals.shape)
	result = predict(out.T, y_test, parameters)
	print(result)
	'''
	X = np.array([24, 3, 35, 45000], dtype=np.float32).reshape(1, 4)
	max_vals = np.array(max_vals, dtype=np.float32)
	X = X / max_vals
	y_test = [1]
	result = classify(X.T, y_test)
	'''
