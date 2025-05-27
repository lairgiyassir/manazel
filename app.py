import streamlit as st
from moroccan_hilal_checker import MoroccanHilalChecker
from hijri_converter import convert
from datetime import datetime, timedelta
import pandas as pd
import io

st.set_page_config(
    page_title="Manazel Project",
    page_icon="🇲🇦",
    layout="wide"
)

# Constants for probability thresholds
LOW_CONFIDENCE_THRESHOLD = 0.8
HIGH_CONFIDENCE_THRESHOLD = 0.9

# For the select box, we need a list of valid Hijri month names.
HIJRI_MONTH_TO_NUMBER = {
    "Muharram": 1,
    "Safar": 2,
    "Rabi' al-awwal": 3,
    "Rabi' al-thani": 4,
    "Jumada al-awwal": 5,
    "Jumada al-thani": 6,
    "Rajab": 7,
    "Sha'ban": 8,
    "Ramadan": 9,
    "Shawwal": 10,
    "Dhu al-Qidah": 11,
    "Dhu al-Hijjah": 12
}

# Get current date and convert to Hijri
current_date = datetime.now()
month_hijri = convert.Gregorian(current_date.year, current_date.month, 1).to_hijri()

def generate_predictions_for_year(hijri_year):
    checker = MoroccanHilalChecker()
    predictions = []
    
    for month_name in HIJRI_MONTH_TO_NUMBER.keys():
        try:
            miladi_year, miladi_month, miladi_day, probability = checker.get_miladi_day_for_hilal(
                hijri_year,
                month_name,
                probability_threshold=HIGH_CONFIDENCE_THRESHOLD
            )
            predictions.append({
                'Hijri Month': month_name,
                'Predicted Date': f"{miladi_year:04d}-{miladi_month:02d}-{miladi_day:02d}",
                'Confidence': f"{probability * 100:.2f}%"
            })
        except Exception as e:
            predictions.append({
                'Hijri Month': month_name,
                'Predicted Date': 'Error',
                'Confidence': str(e)
            })
    
    return pd.DataFrame(predictions)

def main():
    st.title("🇲🇦 Manazel Project")
    st.markdown( "مشروع منازل لتحديد بداية الشهر الهجري في المغرب انطلاقا من حتمالية رؤية الهلال. لا تنسونا من خالص دعائكم")
    st.markdown(
        """
        This application allows you to select a Hijri year and month, then predicts the **first day** 
        of that Hijri month based on a pretrained AI model that estimates the visibility of the hilal in Morocco.
        """
    )
    st.info("Disclaimer : This is an AI prediction and not an official annoucement, please refer to the official authorities for the official date.")

    # User inputs: Hijri year and month
    hijri_year = st.number_input("Hijri Year", min_value=month_hijri.year, max_value=1600, value=month_hijri.year, step=1)
    hijri_months = list(HIJRI_MONTH_TO_NUMBER.keys())
    # Use a selectbox for a dropdown list of valid months
    hijri_month_name = st.selectbox("Hijri Month", hijri_months, index=hijri_months.index(hijri_months[month_hijri.month]))

    
    # Button to trigger computation for single month
    if st.button("Predict the beginning of the month"):
        checker = MoroccanHilalChecker()
        try:
            # Make sure you've already loaded `mor_hilal_vis_model` somewhere,
            # or import it from wherever you defined it.
            miladi_year, miladi_month, miladi_day, probability = checker.get_miladi_day_for_hilal(
                hijri_year, 
                hijri_month_name,
                probability_threshold=LOW_CONFIDENCE_THRESHOLD
            )
            
            if probability >= LOW_CONFIDENCE_THRESHOLD and probability < HIGH_CONFIDENCE_THRESHOLD:
                next_year, next_month, next_day, next_probability = checker.get_miladi_day_for_hilal(
                    hijri_year,
                    hijri_month_name,
                    probability_threshold=HIGH_CONFIDENCE_THRESHOLD
                )
                
                st.warning(
                    f''' ⚠️ This month is tricky! The model predicts {miladi_year:04d}-{miladi_month:02d}-{miladi_day:02d} 
                    with {probability * 100:.2f}% confidence.
                    \nDepending on the moroccan historical confidence rates, consider the next day 
                    ({next_year:04d}-{next_month:02d}-{next_day:02d}) with {next_probability * 100:.2f}% confidence.'''
                )
            else:
                st.success(
                    f"The predicted date for the first {hijri_month_name} {hijri_year} is ➡️ "
                    f"{miladi_year:04d}-{miladi_month:02d}-{miladi_day:02d}, with a confidence of {probability * 100:.2f}%"
                )
        except ValueError as ve:
            st.error(f"ValueError: {ve}")
        except RuntimeError as re:
            st.error(f"RuntimeError: {re}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    
    # Add download button for all months
    if st.button("Download Predictions for All Months For This Year"):
        with st.spinner("Generating predictions for all months..."):
            df = generate_predictions_for_year(hijri_year)
            
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Create download button
            st.download_button(
                label="Download Excel File",
                data=output.getvalue(),
                file_name=f"hilal_predictions_{hijri_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()