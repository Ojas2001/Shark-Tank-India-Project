import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.image("images/WhatsApp-Image-2023-10-18-at-3.25.30-PM-1.jpeg",width=700)


with st.sidebar:
    st.title(f'Shark Tank India')

#st.title("Shark Tank India")
st.header("Deal Amount Prediction")
prediction_model = st.sidebar.selectbox(f'Choose model', ["ANN", "Decision Tree", "Logistic Regression", "Random forest"])

prediction_model_path = {
    "ANN": "models/phase2_models/ANN_phase_2_80-20_split.sav",
    "Decision Tree": "models/phase2_models/Decision_tree_phase_2_80-20_split.sav",
    "Logistic Regression": "models/phase2_models/Logistic_Regression_phase_2_80-20_split.sav",
    "Random forest": "models/phase2_models/Random_forest_phase_2_80-20_split.sav",
}


f = open(prediction_model_path[prediction_model], 'rb')


model_80_20 = pickle.load(f)


def main():
    predict = None
    with st.form(key='Deal'):
        Category = st.selectbox('Category', ("Automative", "Business", "Clothes/Cosmetics", "Consumer Item", "Consumer services", "Food", "Manufacturing good", "Novel Ideas", "Productivity tools", "Technology"), key='Category')
        Male_rating = st.number_input('Male Rating',key='Male Rating',min_value=0,max_value=5)
        Female_rating = st.number_input('Female Rating',key='Female Rating',min_value=0,max_value=5)
        pitcher_ask_amount = st.number_input('Pitcher ask amount (in Lakhs INR)', key='Pitcher ask amount',min_value=0)
        ask_equity = st.number_input('Ask equity (in %)', key='Ask equity',min_value=0,max_value=100)
        ask_valuation = st.number_input('Ask valuation (in Lakhs INR)', key='Ask valuation',min_value=0)
        deal_equity = st.number_input("Deal Equity (in %)",key="Deal Equity",min_value=0,max_value=100)
        deal_valuation = st.number_input("Deal Valuation (in Lakhs INR)",key="Deal Valuation",min_value=0)
        
        amit_deal = st.number_input('Amit deal',min_value=0, key='Amit deal')
        anupam_deal = st.number_input('Anupam deal',min_value=0, key='Anupam deal')
        aman_deal = st.number_input('Aman deal',min_value=0, key='Aman deal')
        namita_deal = st.number_input('Namita deal',min_value=0, key='Namita deal')
        vineeta_deal = st.number_input('Vineeta deal',min_value=0, key='Vineeta deal')
        peyush_deal = st.number_input('Peyush deal',min_value=0, key='Peyush deal')
        ghazal_deal = st.number_input('Ghazal deal',min_value=0, key='Ghazal deal')
        equity_per_shark = st.number_input("Equity per Shark",min_value=0,max_value=100,key="Equity per Shark")

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

                title = ["Category", "Male_rating",	"Female_rating",	"pitcher_ask_amount",	"ask_equity",	"ask_valuation","deal_equity","deal_valuation",	"amit_deal", "anupam_deal", 	"aman_deal",	"namita_deal",	"vineeta_deal",	"peyush_deal",	"ghazal_deal","equity_per_shark"]
                deal_inputs = [Category_mapping[Category], float(Male_rating), float(Female_rating), float(pitcher_ask_amount), float(ask_equity), float(ask_valuation), float(deal_equity), float(deal_valuation),int(amit_deal), int(anupam_deal),int(aman_deal), int(namita_deal), int(vineeta_deal), int(peyush_deal), int(ghazal_deal),float(equity_per_shark)]

                data = pd.read_excel('training.xlsx')

                phase_2_training = data.drop(['deal_amount','amount_per_shark','deal','amit_present','anupam_present',
                              'aman_present','namita_present','vineeta_present','peyush_present',
                              'ghazal_present','total_sharks_invested'], axis=1)
                
                phase_2_scaler = StandardScaler().fit(phase_2_training)
                phase_2_scaler.transform(phase_2_training) 
                deal_inputs = pd.DataFrame([deal_inputs], columns=title)
                deal = phase_2_scaler.transform(deal_inputs) # input for deal amount prediction

                predict_y_80_20 = model_80_20.predict(deal)
                if predict_y_80_20 < 0:
                    predict_y_80_20 = [0]
                try:
                    st.title(f'The deal amount is : {predict_y_80_20[0][0]} Lakhs')
                except:
                    st.title(f'The deal amount is : {predict_y_80_20[0]} Lakhs')

            except Exception as e:
                st.error(f'Error: {e}')


if __name__ == '__main__':
    main()
