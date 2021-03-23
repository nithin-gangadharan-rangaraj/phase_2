import streamlit as st
import streamlit_theme as stt
import numpy as np
import pandas as pd
import pickle
import sklearn

PAGE_CONFIG = {"page_title":"Heart Risk Prediction","page_icon":"💪","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

pickle_in = open('knn.pkl','rb')
model = pickle.load(pickle_in)

st.title("Heart Risk Prediction")

html = '''
<body style="background-color:grey;">
	<h1>Products</h1>
</body>
'''

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


def main():
	st.markdown(
		"""
		<style>
		.reportview-container {
		background: url("https://i.pinimg.com/originals/a0/7e/8f/a07e8f05a7d516ba7fd90519c5126058.jpg")
		background-size: cover;
		}
		</style>
		""", unsafe_allow_html=True) 
	st.markdown(hide_streamlit_style, unsafe_allow_html=True)
	st.subheader("*Enter the following parameters for prediction*")
	age = st.slider("Select your age", 1, 100)
	totChol = st.number_input("Enter Cholesterol")
	sysBP = st.number_input("Enter Systolic BP")
	diaBP = st.number_input("Enter Diastolic BP")
	BMI = st.number_input("Enter BMI")
	heartrate = st.number_input("Enter Heart Rate")
	glucose = st.number_input("Enter Glucose")
	x = [age, totChol, sysBP, diaBP, BMI, heartrate, glucose]
	new = np.array(scale_fun(x))
	if(st.button("Predict")):
		probs = model.predict_proba(new.reshape(1, -1))
		cls = model.predict(new.reshape(1, -1))
		#if (probs[0][1] > probs[0][0]):
		#	percentage = probs[0][1] * 100
		#	st.write("You are ",round(percentage,2),"% at a risk of developing CVD!")
		#elif (probs[0][0] > probs[0][1]):
		#	st.write("No worries, You are safe!")
		if(cls == 1):
			st.write("You are at Risk!")
		elif(cls == 0):
			st.write("You are Safe!")
	
	
def scale_fun(data):
	mean = [ 51.27028395, 240.94786493, 136.17309932,  84.32887391, 26.09129934,  75.99169532,  83.91940731]
	scale = [ 8.40482301, 45.21403725, 23.51018094, 12.35229104,  3.96554568, 11.60813337, 29.02018717]
	transformed_data = []
	for i in range(0,7):
		x = (data[i] - mean[i])/scale[i]
		transformed_data.append(round(x, 8))
	return(transformed_data)	
	
		
		
if __name__ == '__main__':
	main()
