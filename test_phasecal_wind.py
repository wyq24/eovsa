"""Tests for the scheduler's pure wind telemetry decision helper."""

import unittest

from phasecal_wind import evaluate_controller_clocks, evaluate_wind


def _decision(**overrides):
    values = {
        'wind_mph': 10.0,
        'sample_age_seconds': 30.0,
        'sample_time': '2026-09-06 15:09:30',
        'wind_limit_mph': 17.0,
        'scram_state': 0,
        'scram_comm_err': 0,
        'acc_age_seconds': 1.0,
    }
    values.update(overrides)
    return evaluate_wind(**values)


class PhasecalWindDecisionTest(unittest.TestCase):

    def test_calm_valid_telemetry_executes(self):
        result = _decision()
        self.assertFalse(result['skip'])
        self.assertFalse(result['wind_confirmed'])
        self.assertEqual(result['reason'], 'wind within limit')

    def test_limit_is_inclusive(self):
        result = _decision(wind_mph=17.0)
        self.assertTrue(result['skip'])
        self.assertTrue(result['wind_confirmed'])
        self.assertEqual(result['reason'], 'wind at or above limit')

    def test_scram_skips_even_with_lower_average(self):
        result = _decision(wind_mph=3.0, scram_state=1)
        self.assertTrue(result['skip'])
        self.assertTrue(result['wind_confirmed'])
        self.assertTrue(result['scram_reliable'])
        self.assertTrue(result['weather_available'])
        self.assertEqual(result['reason'], 'wind scram active')

    def test_active_scram_skips_when_weather_is_missing(self):
        result = _decision(wind_mph=None, sample_age_seconds=None,
                           scram_state=1)
        self.assertTrue(result['skip'])
        self.assertTrue(result['wind_confirmed'])
        self.assertTrue(result['scram_reliable'])
        self.assertFalse(result['weather_available'])

    def test_controller_clock_check_handles_midnight_rollover(self):
        before_midnight = evaluate_controller_clocks(
            60000.99999, 86399900, 60000, 86399900)
        after_midnight = evaluate_controller_clocks(
            60001.00001, 100, 60001, 100)
        self.assertTrue(before_midnight['available'])
        self.assertTrue(after_midnight['available'])

    def test_controller_clock_check_rejects_future_and_nonfinite_values(self):
        future = evaluate_controller_clocks(
            60000.5, 43200000, 60000, 43202000)
        invalid = evaluate_controller_clocks(
            60000.5, float('nan'), 60000, 43200000)
        self.assertFalse(future['available'])
        self.assertEqual(future['reason'], 'Ant16 controller clock stale')
        self.assertFalse(invalid['available'])
        self.assertEqual(invalid['reason'], 'Ant16 controller clock invalid')

    def test_observed_legacy_windscram_status_is_valid(self):
        result = _decision(scram_comm_err=-1950679035)
        self.assertFalse(result['skip'])
        self.assertFalse(result['wind_confirmed'])

    def test_unrecognized_windscram_error_skips(self):
        result = _decision(scram_comm_err=-1)
        self.assertTrue(result['skip'])
        self.assertEqual(result['reason'],
                         'ACC wind telemetry communication error')

    def test_invalid_weather_is_conservative(self):
        for values in ({'wind_mph': None},
                       {'wind_mph': float('nan')},
                       {'wind_mph': -1.0},
                       {'sample_age_seconds': -1.0},
                       {'sample_age_seconds': 301.0}):
            result = _decision(**values)
            self.assertFalse(result['skip'])
            self.assertFalse(result['wind_confirmed'])
            self.assertFalse(result['weather_available'])
            self.assertEqual(result['reason'], 'weather unavailable')

        result = _decision(wind_limit_mph=float('inf'))
        self.assertTrue(result['skip'])
        self.assertEqual(result['reason'], 'wind limit invalid')

    def test_invalid_or_stale_acc_telemetry_is_conservative(self):
        for values, reason in (
                ({'stateframe_ok': False}, 'ACC stateframe unavailable'),
                ({'acc_age_seconds': -0.1}, 'ACC stateframe stale'),
                ({'acc_age_seconds': 31.0}, 'ACC stateframe stale'),
                ({'scram_state': 0.5}, 'ACC wind telemetry invalid'),
                ({'scram_comm_err': float('inf')}, 'ACC wind telemetry invalid'),
                ({'scram_comm_err': 1},
                 'ACC wind telemetry communication error')):
            result = _decision(**values)
            self.assertTrue(result['skip'])
            self.assertFalse(result['wind_confirmed'])
            self.assertEqual(result['reason'], reason)


if __name__ == '__main__':
    unittest.main()
