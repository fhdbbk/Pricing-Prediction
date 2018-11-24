# Pricing Analytics Solution

In this project, we have proposed a model which predicts the optimum bid price for construction projects taking into account various parameters like geography, risks,vendor details and historical quote price for previous projects. It helps the user to get the best optimum bid price in those particular cases and whether the bid will be successfull or not based on previous data.


### Data : 
We have created a dummy dataset of 2000 rows and 5 columns where each row consist of following paramters:
1. EPC Vendor : Vendor name of the particular project
2. Geography : State in India where the project was developed
3. Historical Quote : Previous quote price of project
4. Risk : Risks associated with the project in percentage
5. Bid Success: Whether the Bid was success or not (0 or 1)

### Solution : 
We have used a linear regression model for predicting the optimum bid price. For predicting whether the Bid will be success or not, we have trained the data through simple neural network. For front end, we have Python Flask framework for getting inputs from the user.

### Results :
Since we have created a dummy dataset which doesn't have any particular pattern to be learned, we got a accuracy of 64%. Below are some of the images of the output:

### Improvement :
Due to inability to find any appropriate real dataset for this problem and shortage of time, we were not able to implement the add extra parameters by the user feature. With the availability of genuine relevant dataset for Indian market, the models will be trained more accurately and predict more accurate results. Thus increasing the chances of a successfull bid.

### Packages needed to run this:
After install Python3, go to your command prompt and install below dependencies:
- pip install Flask
- pip install numpy
- pip install pandas
- pip install sk-learn
- pip install matplotlib

### Running this Project:
Go to the base directory of the project and write
- python app.py

Flask server will start running at http://127.0.0.1:5000/. Open this address in your browser and you are good to go.