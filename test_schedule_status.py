import os
import shutil
import tempfile
import unittest

from schedule_status import build_status_lines, load_executed_lines
from schedule_status import write_schedule_status
from schedule_status import write_executed_lines


class ScheduleStatusTests(unittest.TestCase):
    def test_current_row_takes_precedence_over_skip_markers(self):
        lines = [
            '2026-08-24 15:10:00 ACQUIRE 0319+415',
            '2026-08-24 15:14:00 PHASECAL 0319+415 pcal_hi-all.fsq',
        ]

        output = build_status_lines(
            lines,
            current_index=1,
            executed_lines=lines[1:],
            planned_indices=set([0, 1]),
        )

        self.assertEqual(output, [
            'S ' + lines[0],
            '* ' + lines[1],
        ])

    def test_planned_rows_are_marked_only_when_exported_as_planned(self):
        lines = [
            '2026-08-24 15:10:00 ACQUIRE 0319+415',
            '2026-08-24 15:14:00 PHASECAL 0319+415 pcal_hi-all.fsq',
        ]

        self.assertEqual(
            build_status_lines(lines, planned_indices=set([1])),
            ['  ' + lines[0], 'S ' + lines[1]],
        )
        self.assertEqual(
            build_status_lines(lines),
            ['  ' + lines[0], '  ' + lines[1]],
        )

    def test_executed_rows_stay_marked_when_planned_rows_are_disabled(self):
        lines = [
            '2026-08-24 15:10:00 ACQUIRE 0319+415',
            '2026-08-24 15:14:00 PHASECAL 0319+415 pcal_hi-all.fsq',
        ]

        self.assertEqual(
            build_status_lines(
                lines,
                executed_lines=[lines[1]],
                planned_indices=set(),
            ),
            ['  ' + lines[0], 'S ' + lines[1]],
        )

    def test_sidecar_is_ignored_for_a_different_schedule(self):
        directory = tempfile.mkdtemp()
        try:
            state_path = os.path.join(directory, 'skip_phacal_status.json')
            lines = [
                '2026-08-24 15:10:00 ACQUIRE 0319+415',
                '2026-08-24 15:14:00 PHASECAL 0319+415 pcal_hi-all.fsq',
            ]
            write_executed_lines(state_path, lines, [lines[1]])

            self.assertEqual(load_executed_lines(state_path, lines), [lines[1]])
            self.assertEqual(
                load_executed_lines(state_path, [lines[0] + ' changed', lines[1]]),
                [],
            )
        finally:
            shutil.rmtree(directory)

    def test_status_and_sidecar_export_are_written_together(self):
        directory = tempfile.mkdtemp()
        try:
            status_path = os.path.join(directory, 'status.txt')
            state_path = os.path.join(directory, 'skip_phacal_status.json')
            lines = [
                '2026-08-24 15:10:00 ACQUIRE 0319+415',
                '2026-08-24 15:14:00 PHASECAL 0319+415 pcal_hi-all.fsq',
            ]
            write_schedule_status(
                status_path,
                state_path,
                lines,
                current_index=None,
                executed_lines=[lines[0]],
            )

            with open(status_path) as handle:
                self.assertEqual(handle.read().splitlines(), [
                    'S ' + lines[0],
                    '  ' + lines[1],
                ])
            self.assertEqual(load_executed_lines(state_path, lines), [lines[0]])
            self.assertEqual(
                [name for name in os.listdir(directory) if name.startswith('.')],
                [],
            )
        finally:
            shutil.rmtree(directory)


if __name__ == '__main__':
    unittest.main()
