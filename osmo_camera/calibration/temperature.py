import numpy as np


# ADS1x15 datasheet:
# Adafruit 4-Channel ADC Breakouts https://drive.google.com/open?id=1VpoROFyzoGa0YAAWzQccNExRHWgEfLj2
ADS1115_MAX_VOLTAGE_AT_GAIN_1 = 4.096
ADS1115_BIT_DEPTH = 16
RASPBERRY_PI_VOLTAGE_3_3 = 3.3


def _celcius_to_kelvin(temperature_c):
    return temperature_c + 273.15


def _kelvin_to_celcius(temperature_k):
    return temperature_k - 273.15


# PR103J2 thermistor datasheet:
# https://drive.google.com/open?id=1XYhgVEmt2UQYlyh7GxlQNlW7ZMkUmiaH
T_0_CELCIUS = 25
T_0_KELVIN = _celcius_to_kelvin(T_0_CELCIUS)
R0_PR103J2_DATASHEET = 10000  # Resistance at T0
BETA_PR103J2_DATASHEET = 3892


def thermistor_resistance_given_temperature(
    temperature_c, beta=BETA_PR103J2_DATASHEET, r0=R0_PR103J2_DATASHEET
):
    """Predict thermistor resistance given temperature using the "Beta parameter equation" from
        https://en.wikipedia.org/wiki/Thermistor

    Args:
        temperature_c:
        beta: Optional (defaults to BETA_PR103J2_DATASHEET). Thermistor "Beta parameter"
        r0: Optional (defaults to R0_PR103J2_DATASHEET). Thermistor resistance at T0
    """
    Rinf = r0 * np.exp(-beta / T_0_KELVIN)

    temperature_k = _celcius_to_kelvin(temperature_c)

    return Rinf * np.exp(beta / temperature_k)


def voltage_divider_vout_given_resistances(r2, r1, v_in=RASPBERRY_PI_VOLTAGE_3_3):
    """ Calculate v_out given voltage divider resistors, r1 and r2, and v_in, where the divider is set up:
        |---v_in
        |
        r1
        |
        |---v_out
        |
        r2
        |
        |---Ground

    Args:
        r2: Resistance (in ohms) of r2 in a voltage divider
        r1: Resistance (in ohms) of r1 in a voltage divider
        v_in: Optional (defaults to RASPBERRY_PI_VOLTAGE_3_3). The input voltage to the voltage divider.

    Returns:
        v_out: Output voltage of the voltage divider
    """
    return v_in * (r2 / (r1 + r2))


def digital_count_given_voltage(
    analog_in, v_max=ADS1115_MAX_VOLTAGE_AT_GAIN_1, bit_depth=ADS1115_BIT_DEPTH
):
    """ Calculate the digital output of an ADC given the analog_in and the reference v_max

    Args:
        analog_in: Input analog voltage coming into the ADC
        v_max: Optional (defaults to ADS1115_MAX_VOLTAGE_AT_GAIN_1). Reference max voltage of the ADC
        bit_depth: Optional (defaults ADS1115_BIT_DEPTH). The bit depth of the ADC.
    """
    max_count = (
        2 ** (bit_depth - 1) - 1
    )  # The ADC returns a signed int, so use (bit_depth - 1)

    # Intentionally don't round to an int here, to make simulated math more consistent later
    return (analog_in / v_max) * max_count


def digital_count_given_temperature(
    temperature_c,
    voltage_divider_resistor,
    thermistor_beta=BETA_PR103J2_DATASHEET,
    thermistor_r0=R0_PR103J2_DATASHEET,
    voltage_divider_v_in=RASPBERRY_PI_VOLTAGE_3_3,
    adc_v_max=ADS1115_MAX_VOLTAGE_AT_GAIN_1,
    adc_bit_depth=ADS1115_BIT_DEPTH,
):
    """ Calculate the digital output of an ADC given temperature, using a thermistor in a voltage divider:

        |---v_in
        |
        r1=voltage_divider_resistor
        |
        |---v_out --> analog input into ADC
        |
        r2=thermistor
        |
        |---Ground

        This function is expected to be used to simulate what digital counts we might expect at given temperatures

    Args:
        temperature_c: Temperature in Celcius
        voltage_divider_resistor: Resistance of the r1 resistor in the voltage divider
        thermistor_beta: Optional (defaults to BETA_PR103J2_DATASHEET). Thermistor "Beta parameter".
        thermistor_r0: Optional (defaults to R0_PR103J2_DATASHEET). Thermistor resistance at T0.
        voltage_divider_v_in: Optional (defaults to RASPBERRY_PI_VOLTAGE_3_3). The input voltage to the voltage divider.
        adc_v_max: Optional (defaults to ADS1115_MAX_VOLTAGE_AT_GAIN_1). Reference max voltage of the ADC.
        adc_bit_depth: Optional (defaults to ADS1115_BIT_DEPTH). The bit depth of the ADC.

    Returns:
        digital_count: a signed integer (matching adc_bit_depth) that represents a digital temperature measurement
    """
    thermistor = thermistor_resistance_given_temperature(
        temperature_c, thermistor_beta, thermistor_r0
    )
    v_out = voltage_divider_vout_given_resistances(
        r2=thermistor, r1=voltage_divider_resistor, v_in=voltage_divider_v_in
    )
    digital_count = digital_count_given_voltage(
        analog_in=v_out, v_max=adc_v_max, bit_depth=adc_bit_depth
    )
    return digital_count


def temperature_given_digital_count(
    digital_count,
    voltage_divider_resistor,
    thermistor_beta=BETA_PR103J2_DATASHEET,
    thermistor_r0=R0_PR103J2_DATASHEET,
    voltage_divider_v_in=RASPBERRY_PI_VOLTAGE_3_3,
    adc_v_max=ADS1115_MAX_VOLTAGE_AT_GAIN_1,
    adc_bit_depth=ADS1115_BIT_DEPTH,
):
    """ Calculate the temperature given a measured digital count from an ADC, using a thermistor in a voltage divider:

        |---v_in
        |
        r1=voltage_divider_resistor
        |
        |---v_out --> analog input into ADC
        |
        r2=thermistor
        |
        |---Ground

        This function is expected to be used to generate a specific curve_fit function
        (i.e. with hardcoded constants) to match calibration data.

    Args:
        digital_count: a signed integer (matching adc_bit_depth) that represents a digital temperature measurement
        voltage_divider_resistor: Resistance of the r1 resistor in the voltage divider
        thermistor_beta: Optional (defaults to BETA_PR103J2_DATASHEET). Thermistor "Beta parameter".
        thermistor_r0: Optional (defaults to R0_PR103J2_DATASHEET). Thermistor resistance at T0.
        voltage_divider_v_in: Optional (defaults to RASPBERRY_PI_VOLTAGE_3_3). The input voltage to the voltage divider.
        adc_v_max: Optional (defaults to ADS1115_MAX_VOLTAGE_AT_GAIN_1). Reference max voltage of the ADC.
        adc_bit_depth: Optional (defaults to ADS1115_BIT_DEPTH). The bit depth of the ADC.

    Returns:
        temperature: Calculated temperature in Celcius
    """
    # This function was artisenally reversed from the digital_count_given_temperature function using pencil and paper
    # fmt: off
    numerator = (
        voltage_divider_resistor
        / (thermistor_r0 * np.exp(-thermistor_beta / T_0_KELVIN))
    )
    denominator = (
        ((2 ** (adc_bit_depth - 1) - 1) * voltage_divider_v_in)
        / (digital_count * adc_v_max)
        - 1
    )
    # fmt: on

    temperature_k = thermistor_beta / np.log(numerator / denominator)
    temperature_c = _kelvin_to_celcius(temperature_k)
    return temperature_c


# These constants were calculated using scipy's curve_fit in the jupyter notebook:
# '2019-04-30 Temperature Calibration.ipynb'
VOLTAGE_DIVIDER_RESISTOR_CALIBRATED = 5001.89789
BETA_CALIBRATED = 3906.51696
VOLTAGE_DIVIDER_VIN_CALIBRATED = 3.29490430


def temperature_given_digital_count_calibrated(digital_count):
    return temperature_given_digital_count(
        digital_count,
        voltage_divider_resistor=VOLTAGE_DIVIDER_RESISTOR_CALIBRATED,
        thermistor_beta=BETA_CALIBRATED,
        thermistor_r0=R0_PR103J2_DATASHEET,
        voltage_divider_v_in=VOLTAGE_DIVIDER_VIN_CALIBRATED,
        adc_v_max=ADS1115_MAX_VOLTAGE_AT_GAIN_1,
        adc_bit_depth=ADS1115_BIT_DEPTH,
    )
