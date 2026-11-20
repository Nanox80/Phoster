"""Phase 2: estimate DSSC cell temperature from air temperature and sunlight.

Ross rule: cell temp = air temp + K * sunlight on the cell (W/m²).
"""
from pvlib.temperature import ross

# Published Ross coefficient for a free-standing, well-ventilated panel (K·m²/W).
# Placeholder: refit against measured cell temperatures in Phase 7.
K = 0.0208


def cell_temperature(air_temp_c, sunlight_w_m2, k=K):
    """Cell temperature in °C. Works on single numbers or numpy/pandas arrays."""
    return ross(sunlight_w_m2, air_temp_c, k=k)
