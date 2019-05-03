import pkg_resources

import numpy as np
import pandas as pd
import pytest

from . import temperature as module


PR103J2_temperature_vs_resistance_path = pkg_resources.resource_filename(
    'osmo_camera',
    'calibration/PR103J2_temperature_vs_resistance.csv'
)


class TestThermistorResistanceGivenTemperature:
    def test_returns_R0_at_T0(self):
        actual = module.thermistor_resistance_given_temperature(temperature_c=module.T0_CELCIUS)

        assert actual == module.R0_PR103J2_DATASHEET

    def test_roughly_matches_datasheet_table(self):
        datasheet_table = pd.read_csv(PR103J2_temperature_vs_resistance_path)

        calculated_resistances = datasheet_table['temperature'].apply(module.thermistor_resistance_given_temperature)
        expected_resistances = datasheet_table['resistance']

        np.testing.assert_allclose(expected_resistances, calculated_resistances, rtol=0.02)


class TestVoltageDividerVoutGivenResistances:
    @pytest.mark.parametrize('name, r2, r1, v_in, expected_v_out', [
        ('divides in half', 10, 10, 10, 5),
        ('divides in quarter', 10, 30, 10, 2.5),
    ])
    def test_voltage_divider_vout_given_resistances(self, name, r2, r1, v_in, expected_v_out):
        actual_Vout = module.voltage_divider_vout_given_resistances(r2, r1, v_in)
        assert actual_Vout == expected_v_out

    def test_defaults_to_raspberry_pi_voltage(self):
        actual_Vout = module.voltage_divider_vout_given_resistances(r2=1, r1=1)
        assert actual_Vout == module.RASPBERRY_PI_VOLTAGE / 2


class TestDigitalCountGivenVoltage:
    @pytest.mark.parametrize('analog_in, v_max, expected_digital_count', [
        (-5, 10, -16383.5),
        (5, 10, 16383.5),
        (10, 10, 32767),
    ])
    def test_spot_check_digital_count_given_voltage(self, analog_in, v_max, expected_digital_count):
        actual = module.digital_count_given_voltage(analog_in, v_max, bit_depth=16)
        assert pytest.approx(actual) == expected_digital_count

    def test_defaults_to_adc_vmax(self):
        expected_digital_count = 2**15 - 1
        actual = module.digital_count_given_voltage(analog_in=4.096)
        assert pytest.approx(actual) == expected_digital_count


# Share spot-check test cases between digital_count_given_temperature and temperature_given_digital_count
SPOT_CHECK_TEST_CASES = [
    # temperature, resistor, digital_count
    (25, 10000, 13199.597167),
    (0, 10000, 20263.742126),
    (25, 5000, 17599.462890),
    (0, 5000, 22928.109841),
]


class TestDigitalCountAndTemperatureFunctions:
    @pytest.mark.parametrize('temperature, resistor, expected_digital_count', SPOT_CHECK_TEST_CASES)
    def test_spot_check_digital_count_given_temperature(self, temperature, resistor, expected_digital_count):
        actual = module.digital_count_given_temperature(temperature, voltage_divider_resistor=resistor)
        assert pytest.approx(actual) == expected_digital_count

    @pytest.mark.parametrize('expected_temperature, resistor, digital_count', SPOT_CHECK_TEST_CASES)
    def test_spot_check_temperature_given_digital_count(self, expected_temperature, resistor, digital_count):
        actual = module.temperature_given_digital_count(
            digital_count,
            voltage_divider_resistor=resistor,
        )
        assert pytest.approx(actual, abs=0.01) == expected_temperature

    def test_digital_count_given_temperature_round_trip(self):
        resistor = 10000
        temperatures = pd.Series([0, 7.5, 15, 25, 30, 35])
        digital_counts = module.digital_count_given_temperature(temperatures, voltage_divider_resistor=resistor)
        round_trip_temperatures = module.temperature_given_digital_count(
            digital_counts,
            voltage_divider_resistor=resistor,
        )

        pd.testing.assert_series_equal(temperatures, round_trip_temperatures)
