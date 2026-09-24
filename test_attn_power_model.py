import unittest

import numpy as np

import attn_power_model as model


class AttnPowerModelTest(unittest.TestCase):

    def test_ant15_prediction_preserves_existing_coefficients(self):
        value = model.predict_power_from_attn(
            1, 2, "H", "OFF", env="lab", antenna=15
        )
        expected = -1.0177794117647057 * 3 + 9.843970588235292
        self.assertAlmostEqual(value, expected)

    def test_ant15_replaces_power_above_threshold(self):
        value, replaced = model.replace_power_if_needed(
            -1.0, 1, 2, "H", 0, measured_voltage=1.2, antenna=15
        )
        self.assertTrue(replaced)
        self.assertAlmostEqual(
            value, model.predict_power_from_attn(1, 2, "H", 0, antenna=15)
        )

    def test_legacy_threshold_argument_uses_ant15(self):
        value, replaced = model.replace_power_if_needed(
            -1.0, 1, 2, "H", 0, 1.105, env="lab", measured_voltage=1.2
        )
        self.assertTrue(replaced)
        self.assertAlmostEqual(
            value, model.predict_power_from_attn(1, 2, "H", 0, antenna=15)
        )

    def test_power_below_threshold_passes_through(self):
        value, replaced = model.replace_power_if_needed(
            -3.5, 1, 2, "H", 0, measured_voltage=1.0, antenna=15
        )
        self.assertFalse(replaced)
        self.assertEqual(value, -3.5)

    def test_ant14_calibration_is_absent_or_complete(self):
        configured = (
            14 in model.COEFF_SLOPE,
            14 in model.COEFF_INTERCEPT,
            14 in model.VOLTAGE_THRESHOLD,
        )
        self.assertTrue(all(configured) or not any(configured))

        if not any(configured):
            value, replaced = model.replace_power_if_needed(
                -2.0, 0, 0, "H", 0, measured_voltage=1.2, antenna=14
            )
            self.assertFalse(replaced)
            self.assertEqual(value, -2.0)
            return

        for pol in ("H", "V"):
            self.assertIn(pol, model.VOLTAGE_THRESHOLD[14])
            for nd_state in ("OFF", "ON"):
                self.assertIn(nd_state, model.COEFF_SLOPE[14][pol])
                self.assertIn(
                    nd_state, model.COEFF_INTERCEPT[14]["lab"][pol]
                )

    def test_nan_power_passes_through(self):
        value, replaced = model.replace_power_if_needed(
            np.nan, 0, 0, "H", 0, measured_voltage=1.2, antenna=15
        )
        self.assertFalse(replaced)
        self.assertTrue(np.isnan(value))


if __name__ == "__main__":
    unittest.main()
