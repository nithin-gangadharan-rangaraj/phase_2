import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

PAGE_CONFIG = {"page_title":"Heart Risk Prediction","page_icon":"👨‍⚕️","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

html = '''
<style>
body {
background-image: url("https://img.freepik.com/free-vector/white-elegant-texture-wallpaper_23-2148421854.jpg?size=626&ext=jpg&ga=GA1.2.145878890.1611360000");
background-size: cover;
}
</style>
'''

pickle_in = open('svm_classifier.pkl','rb')
clf_svm = pickle.load(pickle_in)

pickle_in1 = open('full_data','rb')
data = pd.DataFrame(pickle.load(pickle_in1))

st.title("Heart Risk Prediction")

def scale_fun(data):
	mean = [ 51.27028395, 240.94786493, 136.17309932,  84.32887391, 26.09129934,  75.99169532,  83.91940731]
	scale = [ 8.40482301, 45.21403725, 23.51018094, 12.35229104,  3.96554568, 11.60813337, 29.02018717]
	transformed_data = []
	for i in range(0,7):
		x = (data[i] - mean[i])/scale[i]
		transformed_data.append(round(x, 8))
	return(transformed_data)


def main():
	st.markdown(html, unsafe_allow_html=True)
	st.sidebar.title("Sidebar")
	st.sidebar.write("Check the box for predicting *heart risk*")
	if(st.sidebar.checkbox("Risk predictor")):
		st.subheader("*Enter the following parameters for prediction*")
		age = st.number_input("Enter Age")
		totChol = st.number_input("Enter Cholesterol")
		sysBP = st.number_input("Enter Systolic BP")
		diaBP = st.number_input("Enter Diastolic BP")
		BMI = st.number_input("Enter BMI")
		heartrate = st.number_input("Enter Heart Rate")
		glucose = st.number_input("Enter Glucose")
		x = [age, totChol, sysBP, diaBP, BMI, heartrate, glucose]
		new = np.array(scale_fun(x))
		if(st.button("Predict")):
			probs = clf_svm.predict_proba(new.reshape(1, -1))
			if (probs[0][1] > probs[0][0]):
				percentage = probs[0][1] * 100
				st.write("You are ",round(percentage,2),"% at a risk of developing CVD!")
			elif (probs[0][0] > probs[0][1]):
				st.write("No worries, You are safe!")
	
	if(st.sidebar.checkbox("Age analysis")):
		st.subheader("AGE - Statistics")
		positive_cases = data[data['TenYearCHD'] == 1]
		sns.countplot(x='age',data = positive_cases, hue = 'TenYearCHD', palette='husl')
		st.pyplot()
		
if __name__ == '__main__':
	main()
