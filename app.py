import streamlit as st
import streamlit_theme as stt
import numpy as np
import pandas as pd
import pickle
import sklearn

PAGE_CONFIG = {"page_title":"Heart Risk Prediction","page_icon":"💪","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

pickle_in = open('model_final_heart_rank9.pkl','rb')
model = pickle.load(pickle_in)

st.title("Heart Risk Prediction")


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
		background: url("https://i.pinimg.com/originals/a5/91/17/a59117a046cbc0082afe2ce27622c0c4.jpg")
		}
		</style>
		""", unsafe_allow_html=True) 
	st.markdown(hide_streamlit_style, unsafe_allow_html=True)
	st.markdown("**Please *enter the following details* to know your results**")
	age = st.slider("Select your age", 1, 100)
	select_gender = st.selectbox('Select your gender', ['Male', 'Female'])
	if (select_gender == 'Male'):
		male = 1
	elif (select_gender == 'Female'):
		male = 0
	select_hyp = st.selectbox('Are you prevalent to hypertension?', ['Yes', 'No'])
	if (select_hyp == 'Yes'):
		prevalenthyp = 1
	elif (select_hyp == 'No'):
		prevalenthyp = 0
	cigsPerDay = st.number_input("Cigarettes Per Day:")
	totChol = st.number_input("Cholesterol:")
	sysBP = st.number_input("Systolic Blood Pressure:")
	diaBP = st.number_input("Diastolic Blood Pressure:")
	BMI = st.number_input("BMI:")
	heartrate = st.number_input("Heart Rate:")
	glucose = st.number_input("Glucose:")
	x = [male, age, cigsPerDay, prevalenthyp, totChol, sysBP, diaBP, BMI, heartrate, glucose]
	#for parameter in [age, totChol, sysBP, diaBP, BMI, heartrate, glucose]:
	#	if (parameter <= 0.0):
	#		st.markdown("**Please enter valid details!**")
			#check = False
	#		break
	#	else:
			#check = True
	with st.beta_expander("Check Results"):
		if ((age or totChol or sysBP or diaBP or BMI or heartrate or glucose) <= 0.0):
			st.write("Please enter valid details")
		else:
			new = np.array(scale_fun(x))
			probs = model.predict_proba(new.reshape(1, -1))
			cls = model.predict(new.reshape(1, -1))
			result = round((probs[0][1]*100),2)
			st.write("You are at ",result, "% at risk")
	
				
	
			#if (result > 50.0):
			#	st.write("You are at ",result, "% at risk")
			#elif (result <=50.0):
			#	st.write("Don't worry! You are safe 😀")
			#if (probs[0][1] > probs[0][0]):
			#	percentage = probs[0][1] * 100
			#	st.write("You are ",round(percentage,2),"% at risk!")
			#elif (probs[0][0] > probs[0][1]):
			#	st.write("No worries, You are safe!😀")
			#if(cls == 1):
			#	st.write("You are at Risk!")
			#elif(cls == 0):
			#	st.write("You are Safe!")
	
	
def scale_fun(data):
	#mean = [ 51.27028395, 240.94786493, 136.17309932,  84.32887391, 26.09129934,  75.99169532,  83.91940731]
	mean = [  0.49123882,  51.4847255 ,   9.5069302 ,   0.38302923, 240.34093837, 136.7986928 ,  84.61766909,  26.07297567, 76.16238202,  84.07489795]
	#scale = [ 8.40482301, 45.21403725, 23.51018094, 12.35229104,  3.96554568, 11.60813337, 29.02018717]
	scale = [ 0.47121675,  8.38727815, 11.84688265,  0.47195591, 44.88548986, 24.20580684, 12.67850225,  3.9057076 , 11.50550338, 29.51885492]
	transformed_data = []
	for i in range(0,10):
		x = (data[i] - mean[i])/scale[i]
		transformed_data.append(round(x, 8))
	return(transformed_data)	
	
		
		
if __name__ == '__main__':
	main()
