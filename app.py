import matplotlib

matplotlib.use("Agg")

import streamlit as st
from moroccan_hilal_checker import MoroccanHilalChecker
from moroccan_hilal_checker.details import (
    build_details_figure,
    format_knn_vote_paragraph,
    load_dataset,
    neighbours_display_table,
)
from hijri_converter import convert
from datetime import datetime
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


@st.cache_data
def _cached_hilal_dataset():
    return load_dataset()


def _render_knn_neighbour_table(title: str, knn: dict | None) -> None:
    """Show labels of the k closest training rows only (closest first, top row)."""
    if not knn or knn.get("neighbours") is None:
        return
    tbl = neighbours_display_table(knn["neighbours"])
    mixed = knn["votes_1"] > 0 and knn["votes_0"] > 0
    with st.expander(title, expanded=mixed):
        st.dataframe(tbl, hide_index=True, use_container_width=True)


def _render_details_section(
    miladi_year: int,
    miladi_month: int,
    miladi_day: int,
    probability: float,
    hijri_year: int,
    hijri_month_name: str,
) -> None:
    """ARCV/W_topo plot, k-NN summaries, and combined verdict vs. logistic regression."""
    st.subheader("Details")
    st.markdown(
        r"Historical **ARCV** vs **$W_{\mathrm{topo}}$** (Morocco) with your predicted first "
        r"Gregorian day **D** and doubt-nights **D-1** / **D-2**. "
        r"**k = 5** nearest neighbours in normalised feature space vote for class 0 or 1."
    )
    try:
        df_hist = _cached_hilal_dataset()
        fig, det = build_details_figure(
            (miladi_year, miladi_month, miladi_day),
            df_hist,
            hijri_year=hijri_year,
            hijri_month_name=hijri_month_name,
        )
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Could not build details figure: {e}")
        return

    d_str = f"{miladi_year:04d}-{miladi_month:02d}-{miladi_day:02d}"
    st.markdown(
        f"**Logistic regression** (same doubt-night as **D-1**): class **1** (visible), "
        f"probability **{probability * 100:.2f}%**"
    )

    knn_d1 = det.get("knn_d1")
    knn_d2 = det.get("knn_d2")
    d1 = det["d1"]
    d2 = det["d2"]

    if knn_d1 is None:
        st.warning(r"Could not compute Odeh parameters for D-1 (ARCV / $W_{\mathrm{topo}}$ undefined).")

    knn_paragraph = format_knn_vote_paragraph(knn_d1, knn_d2, d1=d1, d2=d2, k=5)
    if knn_paragraph:
        st.markdown(knn_paragraph)

    k_knn = int(det.get("k", 5))
    _render_knn_neighbour_table(
        f"D-1 — {k_knn} closest historical nights (doubt night {d1.isoformat()})",
        knn_d1,
    )
    _render_knn_neighbour_table(
        f"D-2 — {k_knn} closest historical nights ({d2.isoformat()})",
        knn_d2,
    )

    if knn_d2 is None:
        st.caption("D-2: Odeh parameters unavailable for that Gregorian date.")

    logistic_class = 1
    if knn_d1 is not None:
        knn_label = knn_d1["label"]
        if logistic_class == 1 and knn_label == 1:
            st.success(
                f"**Final verdict:** first day of the month **{d_str}** — "
                "logistic regression and k-NN (at D-1) **both** indicate next-day month start (high agreement)."
            )
        elif logistic_class == 1 and knn_label == 0:
            st.warning(
                "**Final verdict (disagreement):** logistic regression favours first day "
                f"**{d_str}**, but k-NN at D-1 suggests the historical **other** class. "
                "Consider treating **the following Gregorian day** as a fallback and consult official announcements."
            )
        elif logistic_class == 0 and knn_label == 0:
            st.info(
                "**Final verdict:** both models suggest class 0 at D-1 (unexpected after a positive stop; treat as a low-confidence edge case)."
            )
        else:
            st.info(
                f"**Final verdict:** logistic class 0 vs k-NN class 1 at D-1 — predicted first day remains **{d_str}**; review manually."
            )
    else:
        st.info("**Final verdict:** k-NN unavailable; rely on the logistic regression message above.")

    if knn_d2 is not None and knn_d2.get("label") == 1:
        st.info(
            "**Note:** k-NN at **D-2** votes for class **1** (next-day start pattern). "
            "The iterative model does not use that night; if you need conservative planning, compare with official calendars."
        )


def main():
    st.title("🇲🇦 Manazel Project")
    st.markdown( "مشروع منازل لتحديد بداية الشهر الهجري في المغرب انطلاقا من احتمالية رؤية الهلال. لا تنسونا من خالص دعائكم")
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
    hijri_month_name = st.selectbox("Hijri Month", hijri_months, index=hijri_months.index(hijri_months[month_hijri.month-1]))

    
    # Button to trigger computation for single month
    if st.button("Predict the beginning of the month"):
        checker = MoroccanHilalChecker()
        try:
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
                    \nLook at the final verdict at the bottom of the page.'''
                )
                st.session_state["last_prediction"] = {
                    "miladi_year": miladi_year,
                    "miladi_month": miladi_month,
                    "miladi_day": miladi_day,
                    "probability": probability,
                    "hijri_year": hijri_year,
                    "hijri_month_name": hijri_month_name,
                }
            else:
                st.success(
                    f"The predicted date for the first {hijri_month_name} {hijri_year} is ➡️ "
                    f"{miladi_year:04d}-{miladi_month:02d}-{miladi_day:02d}, with a confidence of {probability * 100:.2f}%"
                )
                st.session_state["last_prediction"] = {
                    "miladi_year": miladi_year,
                    "miladi_month": miladi_month,
                    "miladi_day": miladi_day,
                    "probability": probability,
                    "hijri_year": hijri_year,
                    "hijri_month_name": hijri_month_name,
                }
        except ValueError as ve:
            st.error(f"ValueError: {ve}")
            st.session_state.pop("last_prediction", None)
        except RuntimeError as re:
            st.error(f"RuntimeError: {re}")
            st.session_state.pop("last_prediction", None)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.pop("last_prediction", None)

    lp = st.session_state.get("last_prediction")
    if (
        lp
        and lp.get("hijri_year") == hijri_year
        and lp.get("hijri_month_name") == hijri_month_name
    ):
        _render_details_section(
            lp["miladi_year"],
            lp["miladi_month"],
            lp["miladi_day"],
            lp["probability"],
            hijri_year,
            hijri_month_name,
        )
    
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