import pandas as pd
from hijri_converter import convert
from utils.odeh import calculate 
import utils.astronomy_ as astronomy
import pickle
from pathlib import Path

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

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / ".." / "models" / "logistic_regression_model.pkl"
MODEL_PATH = MODEL_PATH.resolve()

# Load the model
with open(MODEL_PATH, "rb") as file:
    mor_hilal_vis_model = pickle.load(file)

class MoroccanHilalChecker:
    def __init__(self):
        pass


    def get_miladi_day_for_hilal(self, hijri_year: int, hijri_month_name: str, mor_hilal_vis_model=mor_hilal_vis_model) -> tuple:
        """
        Given a Hijri year and a Hijri month name, returns the corresponding Gregorian date
        for the first day of that Hijri month based on the computed hilal.

        Parameters:
            hijri_year (int): The Hijri year.
            hijri_month_name (str): The name of the Hijri month (e.g., "Ramadan").
            mor_hilal_vis_model: A predictive model that returns a prediction (expected to output 1 when the conditions for hilal are met).

        Returns:
            tuple: A tuple (year, month, day) representing the Gregorian date.
        
        Raises:
            ValueError: If the provided Hijri month name is not valid or if the day cannot be used to estimate crescent visibility.
            RuntimeError: If a valid hilal date cannot be determined within a reasonable number of iterations.
        """
        # Convert the month name to its corresponding number.
        hijri_month = HIJRI_MONTH_TO_NUMBER.get(hijri_month_name)
        if hijri_month is None:
            raise ValueError(f"Invalid Hijri month name: {hijri_month_name}")

        hijri_day = 1

        # Convert the first theoretical Hijri date to a Gregorian date to use it to get the date of 29th of the previous month.
        gregorian_date = convert.Hijri(hijri_year, hijri_month, hijri_day).to_gregorian()

        prediction = 0
        day_offset = -1  # Renamed from add_days for clarity
        max_iterations = 30  # Added safeguard to avoid infinite loops
        iterations = 0

        while prediction != 1 and iterations < max_iterations:
            # Compute the "doubt night" by subtracting day_offset from the computed Gregorian date.
            # We start from the 29th of the previous month to see if the crescent is visible.
            # Otherwise we will see it on the 30th.
            doubt_night = astronomy.Time.Make(
                gregorian_date.year,
                gregorian_date.month,
                gregorian_date.day,
                0, 0, 0
            ).AddDays(day_offset)
            
            # Calculate astronomical parameters needed for prediction. The lat and long are set to Rabat.
            utc_time = doubt_night.Utc()
            parameters = calculate(
                base_time=astronomy.Time.Make(utc_time.year, utc_time.month, utc_time.day, 0, 0, 0),
                latitude=34.0084, 
                longitude=6.8539
            )
            
            # Check if the required keys are present in the parameters.
            if "ARCV" not in parameters or "W_topo" not in parameters:
                raise ValueError("This day is not possible to estimate the crescent visibility for")

            # Prepare data for the model prediction.
            test_df = pd.DataFrame([{"arcv": parameters["ARCV"], "W_topo": parameters["W_topo"]}])
            prediction = int(mor_hilal_vis_model.predict(test_df)[0])
            
            day_offset += 1
            iterations += 1

        if iterations == max_iterations:
            raise RuntimeError("Failed to determine the correct hilal date within the maximum number of iterations.")

        # The first day of the month is the day after the last day the hilal where the hilal is visible.
        first_day_of_the_month = doubt_night.AddDays(1)
        utc_first_day = first_day_of_the_month.Utc()
        return (utc_first_day.year, utc_first_day.month, utc_first_day.day)
