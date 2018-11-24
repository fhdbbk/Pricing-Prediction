from flask import Flask, render_template, flash, request
from predictor import classify, regress
import pickle
import numpy as np

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

STATES = {"Andhra Pradesh" : 0, "Arunachal Pradesh" : 1, "Assam" : 2, "Bihar" : 3, "Chhattisgarh" : 4,
          "Goa": 5, "Gujarat" : 6,"Haryana" : 7,"Himachal Pradesh" : 8, "Jammu and Kashmir" : 9, "Jharkhand" : 10, 
          "Karnataka" : 11, "Kerala" : 12, "Madhya Pradesh" : 13, "Maharashtra" : 14, "Manipur" : 15,"Meghalaya" : 16,
          "Mizoram" : 17, "Nagaland" : 18, "Odisha" : 19, "Punjab" : 20, "Rajasthan" : 21, "Sikkim" : 22,"Tamil Nadu" : 23,
          "Telangana" : 24,"Tripura" : 25, "Uttar Pradesh" : 26, "Uttarakhand" : 27, "West Bengal" : 28}

EPC_VENDORS = {"LNT" : 0, "Nagarjuna" : 1, "GMR" : 2, "Tata" : 3, "Gammon" : 4}

def get_classification(state, vendor, risk, quote):
    '''
    Classifies the quote as successfull or unsuccessfull
    
    Arguments:
    state -- Geography of the Project
    vendor -- EPC vendor
    risk -- Percentage risk involved in the project
    quote -- Predicted quote from our regression model
        
    Returns:
    result -- 0 or 1 telling whether the bid was successfull
    '''
    with open('max_values.pickle', 'rb') as handle:
        max_values = pickle.load(handle)
    max_vals = max_values['max_values']
    print("Max values: {}".format(max_vals))
    print("Type of Max values: {}".format(type(max_vals)))
    X = np.array([EPC_VENDORS[vendor], STATES[state], quote, risk], dtype=np.float32).reshape(1, 4)
    max_vals = np.array(max_vals, dtype=np.float32)
    print("Max values: {}".format(max_vals))
    X = X / max_vals
    y_test = [1]
    result = classify(X.T, y_test)
    return result

def get_prediction(state, vendor, risk):
    X = np.array([EPC_VENDORS[vendor], STATES[state], risk], dtype=np.float32).reshape(1, 3)
    result = regress(X)
    return result    

@app.route("/", methods=['GET', 'POST'])
def hello():
    form = request.form
    STATES = ["Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Goa","Gujarat","Haryana","Himachal Pradesh","Jammu and Kashmir",
    "Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra",
    "Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab",
    "Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal"]
    EPC_VENDORS = ["LNT", "Nagarjuna", "GMR", "Tata", "Gammon"]

    # print(form.errors)
    if request.method == 'POST':
        state=request.form['state']
        vendor=request.form['vendor']
        risk=request.form['risk']
        print(state, " ", vendor, " ", risk)
        predicted_quote = get_prediction(state, vendor, risk)
        classifier_result = get_classification(state, vendor, risk, predicted_quote)
        text = "Optimum Bid Price: {}".format(int(predicted_quote))
        if classifier_result == 1:
            text += ",Bid Successfull"
        else:
            text += ",Bid Unsuccessfull"
        flash('' + text)
        
    return render_template('hello.html', form=form, states=STATES, vendors=EPC_VENDORS)

if __name__ == "__main__":
    app.run(debug = DEBUG)
