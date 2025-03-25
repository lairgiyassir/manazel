import streamlit as st
from moroccan_hilal_checker import MoroccanHilalChecker

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

def main():
    st.title("🇲🇦 Manazel Project - Moroccan Hilal Checker 🇲🇦")
    st.markdown( "مشروع منازل لإحتمالية رؤية الهلال في المغرب لتحديد بداية الشهر الهجري ")
    st.markdown(
        """
        This application allows you to select a Hijri year and month, then predicts the **first day** 
        of that Hijri month based on a pretrained AI model that estimates the visibility of the hilal in Morocco.
        """
    )
    st.markdown(
        """
        Disclaimer : this is an AI prediction and not an official annoucement, please refer to the official authorities for the official date.
        """
    )
    # User inputs: Hijri year and month
    hijri_year = st.number_input("Hijri Year", min_value=1300, max_value=1600, value=1444, step=1)
    hijri_months = list(HIJRI_MONTH_TO_NUMBER.keys())
    # Use a selectbox for a dropdown list of valid months
    hijri_month_name = st.selectbox("Hijri Month", hijri_months, index=hijri_months.index("Ramadan"))

    # Button to trigger computation
    if st.button("Predict the beginning of the month"):
        checker = MoroccanHilalChecker()
        try:
            # Make sure you've already loaded `mor_hilal_vis_model` somewhere,
            # or import it from wherever you defined it.
            miladi_year, miladi_month, miladi_day = checker.get_miladi_day_for_hilal(
                hijri_year, 
                hijri_month_name, 
            )
            st.success(
                f"The computed date for the first {hijri_month_name} - {hijri_year} is : "
                f"{miladi_year:04d}-{miladi_month:02d}-{miladi_day:02d}"
            )
        except ValueError as ve:
            st.error(f"ValueError: {ve}")
        except RuntimeError as re:
            st.error(f"RuntimeError: {re}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()