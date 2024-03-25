import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#st.components.v1.html(custom_html)
st.image("images/WhatsApp-Image-2023-10-18-at-3.25.30-PM-1.jpeg",width=700)


with st.sidebar:
    st.title(f'Shark Tank India')

#st.title("Shark Tank India")
st.header("Deal Prediction")
prediction_model = st.sidebar.selectbox(f'Choose model', ["ANN", "Decision Tree", "Logistic Regression", "Random forest"])

prediction_model_path = {
    "ANN": "models/phase1_models/ANN_final_model.sav",
    "Decision Tree": "models/phase1_models/Decision_tree_phase_1_75-25_split.sav",
    "Logistic Regression": "models/phase1_models/Logistic_Regression_phase_1_75-25_split.sav",
    "Random forest": "models/phase1_models/Random_forest_phase_1_75-25_split.sav",
}


f = open(prediction_model_path[prediction_model], 'rb')

model_75_25 = pickle.load(f)


def main():
    predict = None
    with st.form(key='Deal'):
        Category = st.selectbox('Category', ("Automative", "Business", "Clothes/Cosmetics", "Consumer Item", "Consumer services", "Food", "Manufacturing good", "Novel Ideas", "Productivity tools", "Technology"), key='Category')
        Male_rating = st.number_input('Male Rating', key='Male Rating', min_value=0, max_value=5)
        Female_rating = st.number_input('Female Rating',key='Female Rating',min_value=0,max_value=5)
        pitcher_ask_amount = st.number_input('Pitcher ask amount (in Lakhs INR)', key='Pitcher ask amount',min_value=0)
        ask_equity = st.number_input('Ask equity', key='Ask equity (in %)',min_value=0,max_value=100)
        ask_valuation = st.number_input('Ask valuation', key='Ask valuation (in Lakhs INR)',min_value=0)

        amit_present = st.selectbox('Amit present', ("Yes", "No"), key='Amit present')
        aman_present = st.selectbox('Aman present', ("Yes", "No"), key='Aman present')
        namita_present = st.selectbox('Namita present', ("Yes", "No"), key='Namita present')
        vineeta_present = st.selectbox('Vineeta present', ("Yes", "No"), key='Vineeta present')
        peyush_present = st.selectbox('Peyush present', ("Yes", "No"), key='Peyush present')
        ghazal_present = st.selectbox('Ghazal present', ("Yes", "No"), key='Ghazal present')

        predict = st.form_submit_button('Predict')

        if predict:
            try:
                Category_mapping = {
                    "Automative": 1,
                    "Business": 2,
                    "Clothes/Cosmetics": 3,
                    "Consumer Item": 4,
                    "Consumer services": 5,
                    "Food": 6,
                    "Manufacturing good": 7,
                    "Novel Ideas": 8,
                    "Productivity tools": 9,
                    "Technology": 10
                }

                present_mapping = {
                    "Yes":1,
                    "No":0
                }

                title = ["Category", "Male_rating",	"Female_rating",	"pitcher_ask_amount",	"ask_equity",	"ask_valuation",	"amit_present",	"aman_present",	"namita_present",	"vineeta_present",	"peyush_present",	"ghazal_present"]
                deal_inputs = [Category_mapping[Category], float(Male_rating), float(Female_rating), float(pitcher_ask_amount), float(ask_equity), float(ask_valuation), present_mapping[amit_present], present_mapping[aman_present], present_mapping[namita_present], present_mapping[vineeta_present], present_mapping[peyush_present], present_mapping[ghazal_present]]

                data = pd.read_excel('training.xlsx')

                phase_1_training = data.drop(['deal','deal_amount','deal_equity','deal_valuation','amit_deal','anupam_deal',
                              'aman_deal','namita_deal','vineeta_deal','peyush_deal','ghazal_deal',
                              'total_sharks_invested','equity_per_shark','amount_per_shark',
                              'anupam_present'], axis=1)
                
                phase_1_scaler = StandardScaler().fit(phase_1_training)
                phase_1_scaler.transform(phase_1_training) 
                deal_inputs = pd.DataFrame([deal_inputs], columns=title)
                deal = phase_1_scaler.transform(deal_inputs) # input for deal prediction

                predict_y_75_25 = model_75_25.predict(deal)

                if prediction_model == 'ANN':
                        predict_y_75_25 = 0 if predict_y_75_25 <= 0.5 else 1
                
                if predict_y_75_25 == 0:
                     predict_y_75_25 = "No Deal"
                else:
                     predict_y_75_25 = "Deal"
                
                st.title(f'Prediction result: {predict_y_75_25}')
            except Exception as e:
                st.error(f'Error: {e}')


if __name__ == '__main__':
    main()
