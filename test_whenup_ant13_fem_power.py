"""Tests for automatic Ant 13 FEM power events in solar schedules."""

import unittest

from whenup import add_ant13_fem_power_events, remove_cal


def _line(timestamp, command):
    return timestamp + ' ' + command


def _commands(lines):
    return [line[20:].split()[0] for line in lines]


def _schedule(gap_minutes):
    # Keep the first ACQUIRE separate from the evening ACQUIRE so that the
    # initial warm-up and the optional idle-gap restart are both observable.
    hour = 19 + gap_minutes // 60
    minute = gap_minutes % 60
    evening = '2026-08-21 %02d:%02d:00' % (hour, minute)
    return [
        _line('2026-08-21 09:00:00', 'ACQUIRE morning'),
        _line('2026-08-21 09:20:00', 'SUN'),
        _line('2026-08-21 19:00:00', 'STOW'),
        _line(evening, 'ACQUIRE evening'),
        _line('2026-08-21 23:04:00', 'PHASECAL evening'),
        _line('2026-08-22 00:26:00', 'REWIND'),
    ]


class Ant13FemPowerScheduleTest(unittest.TestCase):

    def test_gap_boundary_and_idempotency(self):
        for gap, expected_on, expected_off in ((9, 1, 1),
                                                (10, 1, 1),
                                                (11, 2, 2)):
            powered = add_ant13_fem_power_events(_schedule(gap))
            self.assertEqual(powered, add_ant13_fem_power_events(powered))
            self.assertEqual([line[:19] for line in powered],
                             sorted(line[:19] for line in powered))
            self.assertEqual(_commands(powered).count('FEMPOWERON'),
                             expected_on)
            self.assertEqual(_commands(powered).count('FEMPOWEROFF'),
                             expected_off)

            if gap > 10:
                stow = _commands(powered).index('STOW')
                off = _commands(powered).index('FEMPOWEROFF')
                restart = _commands(powered).index('FEMPOWERON', stow)
                evening = _commands(powered).index('ACQUIRE', restart)
                self.assertEqual(off, stow + 1)
                self.assertEqual(powered[off][:19], powered[stow][:19])
                self.assertEqual(powered[restart][:19],
                                 '2026-08-21 19:06:00')
                self.assertLess(restart, evening)

    def test_earlier_stow_does_not_trigger_evening_cycle(self):
        lines = [
            _line('2026-08-21 07:30:00', 'ACQUIRE morning'),
            _line('2026-08-21 08:55:00', 'STOW'),
            _line('2026-08-21 09:00:00', 'SUN'),
            _line('2026-08-21 19:00:00', 'ACQUIRE evening'),
            _line('2026-08-21 20:26:00', 'REWIND'),
        ]
        powered = add_ant13_fem_power_events(lines)
        self.assertEqual(_commands(powered).count('FEMPOWERON'), 1)
        self.assertEqual(_commands(powered).count('FEMPOWEROFF'), 1)

    def test_no_evening_acquire_powers_off_at_final_stow(self):
        lines = [
            _line('2026-08-21 09:00:00', 'SUN'),
            _line('2026-08-21 19:00:00', 'STOW'),
            _line('2026-08-21 19:01:00', 'REWIND'),
        ]
        powered = add_ant13_fem_power_events(lines)
        self.assertEqual(_commands(powered),
                         ['FEMPOWERON', 'SUN', 'STOW', 'FEMPOWEROFF',
                          'REWIND'])
        self.assertEqual(powered, add_ant13_fem_power_events(powered))

    def test_remove_cal_keeps_rewind_after_no27m_fem_shutdown(self):
        lines = [
            _line('2026-08-21 09:00:00', 'ACQUIRE morning'),
            _line('2026-08-21 09:20:00', 'SUN'),
            _line('2026-08-21 19:00:00', 'STOW'),
            _line('2026-08-21 23:00:00', 'ACQUIRE evening'),
            _line('2026-08-21 23:04:00', 'PHASECAL evening'),
            _line('2026-08-22 00:26:00', 'REWIND'),
        ]
        no27m = remove_cal(lines, ant13_fem_power=True)
        self.assertEqual(_commands(no27m),
                         ['FEMPOWERON', 'SUN', 'STOW', 'FEMPOWEROFF',
                          'REWIND'])
        self.assertEqual(no27m[-1][:19], '2026-08-21 19:01:00')

    def test_remove_cal_does_not_use_morning_stow_as_solar_end(self):
        lines = [
            _line('2026-08-21 07:30:00', 'ACQUIRE morning'),
            _line('2026-08-21 08:55:00', 'STOW'),
            _line('2026-08-21 09:00:00', 'SUN'),
            _line('2026-08-21 19:00:00', 'ACQUIRE evening'),
            _line('2026-08-21 19:04:00', 'PHASECAL evening'),
            _line('2026-08-21 20:26:00', 'REWIND'),
        ]
        no27m = remove_cal(lines, ant13_fem_power=True)
        self.assertEqual(_commands(no27m),
                         ['FEMPOWERON', 'SUN', 'FEMPOWEROFF', 'REWIND'])
        self.assertEqual(no27m[-2][:19], '2026-08-21 19:00:00')
        self.assertEqual(no27m[-1][:19], '2026-08-21 19:01:00')


if __name__ == '__main__':
    unittest.main()
