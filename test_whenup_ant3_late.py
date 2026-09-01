"""Tests for Ant 3 late-start macro handling in remove_cal schedules."""

import unittest

from whenup import remove_cal


def _line(timestamp, command):
    return timestamp + ' ' + command


def _commands(lines):
    return [line[20:].split()[0] for line in lines]


class Ant3LateStartScheduleTest(unittest.TestCase):

    def test_ant3_late_start_schedule(self):
        evening_acquire = '2026-09-01 01:48:00'
        lines = [
            _line('2026-08-31 12:19:00', 'ACQUIRE 2253+161'),
            _line('2026-08-31 12:20:00', 'LOSELECT'),
            _line('2026-08-31 12:23:00',
                  'PHASECAL_LO 2253+161 pcal_lo.fsq'),
            _line('2026-08-31 12:43:00', 'HISELECT'),
            _line('2026-08-31 12:44:00',
                  'PHASECAL 2253+161 pcal_hi-all.fsq'),
            _line('2026-08-31 13:40:00', 'STOW'),
            _line('2026-08-31 14:18:00', 'SUN_NO_ANT3'),
            _line('2026-08-31 14:45:00', 'SUN_ANT3'),
            _line('2026-08-31 15:10:00', 'ACQUIRE 2253+161'),
            _line('2026-08-31 15:14:00',
                  'PHASECAL 2253+161 pcal_hi-all.fsq'),
            _line('2026-08-31 15:30:00', 'SUN'),
            _line('2026-08-31 18:30:00', 'SOLPNTCAL solar.fsq'),
            _line('2026-08-31 18:37:00', 'GAINSOLPNT'),
            _line('2026-08-31 18:39:00', 'SUN'),
            _line('2026-08-31 19:50:00', 'ACQUIRE 0319+415'),
            _line('2026-08-31 19:54:00',
                  'PHASECAL 0319+415 pcal_hi-all.fsq'),
            _line('2026-08-31 20:10:00', 'SUN'),
            _line('2026-09-01 01:20:00', 'STOW_ANT3'),
            _line('2026-09-01 01:47:00', 'STOW'),
            _line(evening_acquire, 'ACQUIRE 1229+020'),
            _line('2026-09-01 01:49:00', 'LOSELECT'),
            _line('2026-09-01 01:52:00',
                  'PHASECAL_LO 1229+020 pcal_lo.fsq'),
            _line('2026-09-01 02:12:00', 'HISELECT'),
            _line('2026-09-01 02:13:00',
                  'PHASECAL 1229+020 pcal_hi-all.fsq'),
            _line('2026-09-01 03:35:00', 'REWIND'),
        ]
        no27m = remove_cal(lines)
        self.assertEqual(_commands(no27m),
                         ['SUN_NO_ANT3', 'SUN_ANT3', 'SOLPNTCAL',
                          'GAINSOLPNT', 'SUN', 'STOW_ANT3', 'STOW',
                          'REWIND'])
        self.assertEqual(_commands(no27m).count('SUN_ANT3'), 1)
        self.assertEqual(no27m[-1][:19], evening_acquire)

    def test_legacy_schedule_unchanged(self):
        evening_acquire = '2026-09-01 01:48:00'
        lines = [
            _line('2026-08-31 12:19:00', 'ACQUIRE 2253+161'),
            _line('2026-08-31 12:20:00', 'LOSELECT'),
            _line('2026-08-31 12:23:00',
                  'PHASECAL_LO 2253+161 pcal_lo.fsq'),
            _line('2026-08-31 12:43:00', 'HISELECT'),
            _line('2026-08-31 12:44:00',
                  'PHASECAL 2253+161 pcal_hi-all.fsq'),
            _line('2026-08-31 13:40:00', 'STOW'),
            _line('2026-08-31 14:18:00', 'SUN'),
            _line('2026-08-31 15:10:00', 'ACQUIRE 2253+161'),
            _line('2026-08-31 15:14:00',
                  'PHASECAL 2253+161 pcal_hi-all.fsq'),
            _line('2026-08-31 15:30:00', 'SUN'),
            _line('2026-08-31 18:30:00', 'SOLPNTCAL solar.fsq'),
            _line('2026-08-31 18:37:00', 'GAINSOLPNT'),
            _line('2026-08-31 18:39:00', 'SUN'),
            _line('2026-08-31 19:50:00', 'ACQUIRE 0319+415'),
            _line('2026-08-31 19:54:00',
                  'PHASECAL 0319+415 pcal_hi-all.fsq'),
            _line('2026-08-31 20:10:00', 'SUN'),
            _line('2026-09-01 01:47:00', 'STOW'),
            _line(evening_acquire, 'ACQUIRE 1229+020'),
            _line('2026-09-01 01:49:00', 'LOSELECT'),
            _line('2026-09-01 01:52:00',
                  'PHASECAL_LO 1229+020 pcal_lo.fsq'),
            _line('2026-09-01 02:12:00', 'HISELECT'),
            _line('2026-09-01 02:13:00',
                  'PHASECAL 1229+020 pcal_hi-all.fsq'),
            _line('2026-09-01 03:35:00', 'REWIND'),
        ]
        no27m = remove_cal(lines)
        self.assertEqual(_commands(no27m),
                         ['SUN', 'SOLPNTCAL', 'GAINSOLPNT', 'SUN', 'STOW',
                          'REWIND'])
        self.assertEqual(no27m[-1][:19], evening_acquire)


if __name__ == '__main__':
    unittest.main()
