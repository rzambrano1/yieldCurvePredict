## -- REQUIRED LIBRARIES -- ##
import streamlit as st

st.set_page_config(page_title='Models')

## -- -- ##

def welcome_page():
    st.title("Welcome to Models!")
    st.write("""Click on the model name to load and run the model""")
    st.write("""**Beware, the loading and fitting time of some models may take up to 15 minutes**""")
    if st.button(label='Yield Curve Prediction with XGBoost Model'):
        with st.spinner('Fetching the data... fitting the model... predicting...'):
            #import time
            #time.sleep(5)
            from predict_page import show_predict_page
            #show_predict_page()
            st.balloons()
        st.success('Done!')

welcome_page()




#from predict_page import show_predict_page
