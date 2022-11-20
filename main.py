import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

RFregressor = pickle.load(open('fitted_model_2-0-RFR.pickle', 'rb'))
LRregressor=pickle.load(open('fitted_model_2-0-LR.pickle', 'rb'))
SVRregressor=pickle.load(open('fitted_model_2-0-SVR.pickle', 'rb'))
XGboostRegressor=pickle.load(open('fitted_model_2-0-XGboost.pickle', 'rb'))

def predict_val(Distance, Temperature, Wind_Chill, Humidity, Pressure, Visibility, Wind_Speed, Precipitation, Bump,
                Crossing, Roundabout, Stop, Traffic_Signal, Accident_year, Accident_month, Accident_hour):
    df = pd.read_csv('NEW BUILD transformed-2-0-csv.csv')
    df = df.drop(['Probability'], axis=1)
    num_arr = pd.DataFrame(np.array([[Distance, Temperature, Wind_Chill, Humidity, Pressure, Visibility, Wind_Speed,
                                      Precipitation, Bump, Crossing, Roundabout, Stop, Traffic_Signal, Accident_year,
                                      Accident_month, Accident_hour]]), columns=df.columns)
    df_concatnd = pd.concat([df, num_arr], axis=0)
    df_concatnd.reset_index(drop=True, inplace=True)
    df_concatnd_last = df_concatnd.iloc[-1, :]
    prediction_RFR = RFregressor.predict(np.array([df_concatnd_last]))
    prediction_LR=LRregressor.predict(np.array([df_concatnd_last]))
    prediction_SVR=SVRregressor.predict(np.array([df_concatnd_last]))
    prediction_XGB=XGboostRegressor.predict(np.array([df_concatnd_last]))
    probability_array=[prediction_RFR,prediction_LR,prediction_SVR,prediction_XGB]
    final_probabilty_array = [ np.round(i, 5) * 100 for i in probability_array]
    return final_probabilty_array
def plot_g(chances):


    labels = 'Accident', 'NO Accident'
    sizes = [chances[0],100-chances[0]]
    explode = (0, 0.1)  # only "explode" the 2nd slice

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)




def UI():

    st.title("Road Safety -  Accident Probabilty Prediction")


    Distance = st.number_input('Distance Affected Due to Traffic (mi)',min_value=0.0)
    col1, col2, col3 = st.columns(3)
    Temperature = col2.number_input('Temperature(F)',min_value=0.0)
    Wind_Chill = col3.number_input('Wind Chill(F)',min_value=0.0)
    Humidity = col1.number_input('Humidity(%)',min_value=0.0)
    Pressure = col2.number_input('Pressure(in)',min_value=0.0)
    Visibility = col3.number_input('Visibility(mi)',min_value=0.0)

    # Wind_Direction=col1.text_input('Wind_Direction') #object

    Wind_Speed = col2.number_input('Wind_Speed(mph)',min_value=0.0)
    Precipitation = col3.number_input('Precipitation(in)',min_value=0.0)

    # Weather_Condition=col1.text_input('Weather_Condition') #object

    Bump = col2.radio('Bump', [True, False])
    Crossing = col3.radio('Crossing', [True, False])
    Roundabout = col2.radio('Roundabout', [True, False])

    Traffic_Signal = col3.radio('Traffic Signal', [True, False])

    # Sunrise_Sunset=col1.text_input('Sunrise_Sunset') #object

    Accident_year = col1.number_input('Accident Year(YYYY)',min_value=2020,value=2021)
    Accident_month = col1.number_input('Accident Month(MM)',min_value=1,max_value=12)
    Accident_hour = col1.number_input('Accident Date(DD)',min_value=1,max_value=31)
    Stop = col1.radio('Stop', [True, False])
    if col2.button('Predict Accident Chances'):
        predicted_probability = predict_val(Distance, Temperature, Wind_Chill, Humidity, Pressure, Visibility, Wind_Speed,
                                         Precipitation, Bump,
                                         Crossing, Roundabout, Stop, Traffic_Signal, Accident_year, Accident_month,
                                         Accident_hour)
        col2.success( 'Accident Chances: {} %'.format(predicted_probability[0]))



        st.write("Using RandomForest Regressor(Current Model)")
        plot_g(abs(predicted_probability[0]))
        st.write("Using Linear Regression")
        plot_g(abs(predicted_probability[1]))
        st.write("Using SVR")
        plot_g(abs(predicted_probability[2]))
        st.write("Using XGBoost")
        plot_g(abs(predicted_probability[3]))




UI()

