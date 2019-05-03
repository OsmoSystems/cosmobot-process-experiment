import numpy as np


ADS1115_MAX_VOLTAGE_AT_GAIN_1 = 4.096
ADS1115_BIT_DEPTH = 16
RASPBERRY_PI_VOLTAGE = 3.3


def _celcius_to_kelvin(temperature_C):
    return temperature_C + 273.15


def _kelvin_to_celcius(temperature_K):
    return temperature_K - 273.15


# PR103J2 thermistor datasheet:
# https://drive.google.com/open?id=1XYhgVEmt2UQYlyh7GxlQNlW7ZMkUmiaH
T0_CELCIUS = 25
T0_KELVIN = _celcius_to_kelvin(T0_CELCIUS)
R0_PR103J2_DATASHEET = 10000  # Resistance at T0
BETA_PR103J2_DATASHEET = 3892


def thermistor_resistance_given_temperature(temperature_C, B=BETA_PR103J2_DATASHEET, R0=R0_PR103J2_DATASHEET):
    '''Predict thermistor resistance given temperature using the "Beta parameter equation" from
        https://en.wikipedia.org/wiki/Thermistor

    Args:
        temperature_C:
        B: Optional (defaults to BETA_PR103J2_DATASHEET). Thermistor "Beta parameter"
        R0: Optional (defaults to R0_PR103J2_DATASHEET). Thermistor resistance at T0
    '''
    Rinf = R0 * np.exp(-B / T0_KELVIN)

    temperature_K = _celcius_to_kelvin(temperature_C)

    return Rinf * np.exp(B / temperature_K)


def voltage_divider_vout_given_resistances(R2, R1, Vin=RASPBERRY_PI_VOLTAGE):
    ''' Calculate Vout given voltage divider resistors, R1 and R2, and Vin, where the divider is set up:
        |---Vin
        |
        R1
        |
        |---Vout
        |
        R2
        |
        |---Ground

    Args:
        R2: Resistance (in ohms) of R2 in a voltage divider
        R1: Resistance (in ohms) of R1 in a voltage divider
        Vin: Optional (defaults to RASPBERRY_PI_VOLTAGE). The input voltage to the voltage divider.

    Returns:
        Vout: Output voltage of the voltage divider
    '''
    return Vin * (R2 / (R1 + R2))


def digital_count_given_voltage(analog_voltage, Vmax=ADS1115_MAX_VOLTAGE_AT_GAIN_1, bit_depth=ADS1115_BIT_DEPTH):
    ''' Calculate the digital output of an ADC given the analog_voltage and the reference Vmax

    Args:
        analog_voltage: Input analog voltage coming into the ADC
        Vmax: Optional (defaults to ADS1115_MAX_VOLTAGE_AT_GAIN_1). Reference max voltage of the ADC
        bit_depth: Optional (defaults ADS1115_BIT_DEPTH). The bit depth of the ADC.
    '''
    max_count = 2**(bit_depth-1) - 1

    # Intentionally don't round to an int here, to make simulated math more consistent later
    return (analog_voltage / Vmax) * max_count


def digital_count_given_temperature(
    temperature_C,
    voltage_divider_resistor,
    thermistor_B=BETA_PR103J2_DATASHEET,
    thermistor_R0=R0_PR103J2_DATASHEET,
    voltage_divider_Vin=RASPBERRY_PI_VOLTAGE,
    adc_Vmax=ADS1115_MAX_VOLTAGE_AT_GAIN_1,
    adc_bit_depth=ADS1115_BIT_DEPTH
):
    ''' Calculate the digital output of an ADC given temperature, using a thermistor in a voltage divider:

        |---Vin
        |
        R1=voltage_divider_resistor
        |
        |---Vout --> analog input into ADC
        |
        R2=thermistor
        |
        |---Ground

        This function is expected to be used to simulate what digital counts we might expect at given temperatures

    Args:
        temperature_C: Temperature in Celcius
        voltage_divider_resistor: Resistance of the R1 resistor in the voltage divider
        thermistor_B: Optional (defaults to BETA_PR103J2_DATASHEET). Thermistor "Beta parameter".
        thermistor_R0: Optional (defaults to R0_PR103J2_DATASHEET). Thermistor resistance at T0.
        voltage_divider_Vin: Optional (defaults to RASPBERRY_PI_VOLTAGE). The input voltage to the voltage divider.
        adc_Vmax: Optional (defaults to ADS1115_MAX_VOLTAGE_AT_GAIN_1). Reference max voltage of the ADC.
        adc_bit_depth: Optional (defaults ADS1115_BIT_DEPTH). The bit depth of the ADC.

    Returns:
        digital_count: a signed integer (matching adc_bit_depth) that represents a digital temperature measurement
    '''
    thermistor = thermistor_resistance_given_temperature(temperature_C, thermistor_B, thermistor_R0)
    Vout = voltage_divider_vout_given_resistances(R2=thermistor, R1=voltage_divider_resistor, Vin=voltage_divider_Vin)
    digital_count = digital_count_given_voltage(analog_voltage=Vout, Vmax=adc_Vmax, bit_depth=adc_bit_depth)
    return digital_count


def temperature_given_digital_count(
    digital_count,
    voltage_divider_resistor,
    thermistor_B=BETA_PR103J2_DATASHEET,
    thermistor_R0=R0_PR103J2_DATASHEET,
    voltage_divider_Vin=RASPBERRY_PI_VOLTAGE,
    adc_Vmax=ADS1115_MAX_VOLTAGE_AT_GAIN_1,
    adc_bit_depth=ADS1115_BIT_DEPTH
):
    ''' Calculate the temperature given a measured digital count from an ADC, using a thermistor in a voltage divider:

        |---Vin
        |
        R1=voltage_divider_resistor
        |
        |---Vout --> analog input into ADC
        |
        R2=thermistor
        |
        |---Ground

        This function is expected to be used to generate a specific curve_fit function
        (i.e. with hardcoded constants) to match calibration data.

    Args:
        digital_count: a signed integer (matching adc_bit_depth) that represents a digital temperature measurement
        voltage_divider_resistor: Resistance of the R1 resistor in the voltage divider
        thermistor_B: Thermistor "Beta parameter".
        thermistor_R0: Thermistor resistance at T0.
        voltage_divider_Vin: The input voltage to the voltage divider.
        adc_Vmax: Reference max voltage of the ADC.
        adc_bit_depth: Optional (defaults ADS1115_BIT_DEPTH). The bit depth of the ADC.

    Returns:
        temperature: Calculated temperature in Celcius
    '''
    # This function was artisenally reversed from the digital_count_given_temperature function using pencil and paper
    numerator = (voltage_divider_resistor) / ((thermistor_R0 * np.exp(-thermistor_B / T0_KELVIN)))
    denominator = ((2**(adc_bit_depth-1) - 1) * voltage_divider_Vin) / (digital_count * adc_Vmax) - 1
    temperature_K = thermistor_B / np.log(numerator / denominator)
    temperature_C = _kelvin_to_celcius(temperature_K)
    return temperature_C


# These constants were calculated using scipy's curve_fit in the jupyter notebook:
# '2019-04-30 Temperature Calibration.ipynb'
VOLTAGE_DIVIDER_RESISTOR_CALIBRATED = 5001.89789
BETA_CALIBRATED = 3906.51696
VOLTAGE_DIVIDER_VIN_CALIBRATED = 3.29490430


def temperature_given_digital_count_calibrated(digital_count):
    return temperature_given_digital_count(
        digital_count,
        voltage_divider_resistor=VOLTAGE_DIVIDER_RESISTOR_CALIBRATED,
        thermistor_B=BETA_CALIBRATED,
        thermistor_R0=R0_PR103J2_DATASHEET,
        voltage_divider_Vin=VOLTAGE_DIVIDER_VIN_CALIBRATED,
        adc_Vmax=ADS1115_MAX_VOLTAGE_AT_GAIN_1,
        adc_bit_depth=ADS1115_BIT_DEPTH
    )
