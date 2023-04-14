import numpy as np
import pandas as pd
import streamlit as st
import pickle
import calendar

st.set_page_config(layout="centered", page_title="Time-series Sales Prediction",
                    menu_items={
                                "Get Help": "https://www.linkedin.com/in/andik-al-fauzi/",
                                "Report a bug": "https://github.com/andik-alfauzi",
                                "About": "### Time-series Sales Prediction App - By Andik Al Fauzi"})

# load the model 
with open('SVMModel.pkl', 'rb') as file1:
  SVMModel = pickle.load(file1)

with open('Scaling.pkl', 'rb') as file1:
  scaler = pickle.load(file1)

# Define main page
def run():
    # Read dataset
    data = pd.read_csv('https://raw.githubusercontent.com/andik-alfauzi/Final-Project/main/sample_dataset_timeseries_noarea.csv')
    data = data.groupby('week_start_date', as_index=False)['quantity'].sum()
    st.dataframe(data)

    # Change into datetime
    data['week_start_date'] = pd.to_datetime(data['week_start_date'], format='%Y-%m-%d')

    # Create a dataframe
    sales = data.groupby('week_start_date')['quantity'].sum()

    # Create A New Dataset with `window=4`
    window = 4
    X = []
    y = []

    for index in range(0, len(sales)-window):
        X.append(sales[index : window + index])
        y.append(sales[window + index])

    X = np.array(X)
    y = np.array(y)

    with st.form(key='time-series-prediction'):
        # Button submit
        submitted = st.form_submit_button('Predict 4 Next Week Sales')

    if submitted:
       # Define function forcasting
        def forecasting(week):
            sales_forecast = sales.copy()
            window = 4
            for i in range(week):
                X = np.array(sales_forecast[-window:].values).reshape(1, -1)
                X_scaled = scaler.transform(X)

                # add  7 last day into dataset
                last_date = sales_forecast.index[-1]
                new_date = last_date + pd.Timedelta(days=7)

                # make sure the date are valid
                while True:
                    _, last_day = calendar.monthrange(new_date.year, new_date.month)
                    if new_date.day <= last_day:
                        break
                    new_date -= pd.Timedelta(days=1)

                sales_forecast[new_date] = round(SVMModel.predict(X_scaled)[0])

            return sales_forecast
        
        # Forecasting sales for the Next 4 weeks
        sales_forecast = forecasting(4)
        
        # Displaying forecast
        st.dataframe(sales_forecast.tail(4))

if __name__ == '__main__':
   run()