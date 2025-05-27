import pandas as pd
from hijri_converter import convert
from utils.odeh import calculate 
import utils.astronomy_ as astronomy
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict

HIJRI_MONTH_TO_NUMBER: Dict[str, int] = {
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

class MoroccanHilalChecker:
    """A class to check for the visibility of the new moon (hilal) in Morocco.
    
    This class uses astronomical calculations and a machine learning model to predict
    the visibility of the new moon, which is crucial for determining the start of
    Islamic months.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the MoroccanHilalChecker.
        
        Args:
            model_path: Optional path to the machine learning model file. If not provided,
                       uses the default model path.

        """
        self.model_path = model_path or MODEL_PATH
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the machine learning model for hilal visibility prediction."""
        try:
            with open(self.model_path, "rb") as file:
                self.mor_hilal_vis_model = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def get_miladi_day_for_hilal(
        self,
        hijri_year: int,
        hijri_month_name: str,
        mor_hilal_vis_model: Optional[object] = None,
        probability_threshold: Optional[float] = 0.9
    ) -> Tuple[int, int, int, float]:
        """Calculate the Gregorian date for the first day of a Hijri month based on hilal visibility.
        
        This method uses astronomical calculations and a machine learning model to determine
        when the new moon (hilal) will be visible, which marks the start of a new Hijri month.
        
        Args:
            hijri_year: The Hijri year.
            hijri_month_name: The name of the Hijri month (e.g., "Ramadan").
            mor_hilal_vis_model: Optional custom model for hilal visibility prediction.
                                If not provided, uses the default model.
            probability_threshold: Optional custom probability threshold for visibility.
                                 If not provided, uses the instance's threshold.
        
        Returns:
            A tuple containing:
            - year (int): The Gregorian year
            - month (int): The Gregorian month (1-12)
            - day (int): The Gregorian day (1-31)
            - probability (float): The probability of hilal visibility
        
        Raises:
            ValueError: If the provided Hijri month name is invalid
            RuntimeError: If a valid hilal date cannot be determined within the maximum iterations
            FileNotFoundError: If the model file cannot be found
            Exception: For other unexpected errors during calculation
        """
        # Use instance values if not provided
        mor_hilal_vis_model = mor_hilal_vis_model or self.mor_hilal_vis_model

        # Validate Hijri month
        hijri_month = HIJRI_MONTH_TO_NUMBER.get(hijri_month_name)
        if hijri_month is None:
            raise ValueError(f"Invalid Hijri month name: {hijri_month_name}")

        hijri_day = 1
        max_iterations = 30
        iterations = 0
        day_offset = -1

        try:
            # Convert the first theoretical Hijri date to Gregorian
            gregorian_date = convert.Hijri(hijri_year, hijri_month, hijri_day).to_gregorian()
            
            while iterations < max_iterations:
                # Calculate the "doubt night" (29th of previous month)
                doubt_night = astronomy.Time.Make(
                    gregorian_date.year,
                    gregorian_date.month,
                    gregorian_date.day,
                    0, 0, 0
                ).AddDays(day_offset)
                
                # Calculate astronomical parameters for Rabat
                utc_time = doubt_night.Utc()
                parameters = calculate(
                    base_time=astronomy.Time.Make(utc_time.year, utc_time.month, utc_time.day, 0, 0, 0),
                    latitude=34.0084,  # Rabat coordinates
                    longitude=6.8539
                )
                
                # Validate required parameters
                if "ARCV" not in parameters or "W_topo" not in parameters:
                    day_offset += 1
                    iterations += 1
                    continue

                # Make prediction using the model
                test_df = pd.DataFrame([{
                    "arcv": parameters["ARCV"],
                    "W_topo": parameters["W_topo"]
                }])
                
                prediction = int(mor_hilal_vis_model.predict(test_df)[0])
                probability = mor_hilal_vis_model.predict_proba(test_df)[0][1]

                if prediction == 1 and probability >= probability_threshold:
                    # First day of the month is the day after the last day the hilal is visible
                    first_day_of_the_month = doubt_night.AddDays(1)
                    utc_first_day = first_day_of_the_month.Utc()
                    return (utc_first_day.year, utc_first_day.month, utc_first_day.day, probability)

                day_offset += 1
                iterations += 1

            raise RuntimeError(
                f"Failed to determine the correct hilal date within {max_iterations} iterations. "
                f"Last calculated probability: {probability:.2f}"
            )

        except Exception as e:
            raise RuntimeError(f"Error calculating hilal date: {str(e)}")
