"""Scheduler-native progression tests for automatic wind pair skipping.

The production scheduler is a Tk application with legacy Python 2 imports.
These tests extract only its scheduling methods from the current source and
run them against small Tk-like list/status doubles.  This exercises the same
transition methods used by ``inc_time`` without opening a display or sending
commands to ACC.
"""

import ast
import copy
from datetime import datetime
from datetime import timedelta
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
import os
import sys
import tempfile
import unittest
import json
import numpy

from phasecal_wind import evaluate_controller_clocks
from schedule_status import load_executed_lines, write_executed_lines


_EPOCH = datetime(1970, 1, 1)
_NOW = [_EPOCH]


def _mjd(line=None):
    value = _NOW[0] if line is None else datetime.strptime(
        line[:19], '%Y-%m-%d %H:%M:%S')
    return (value - _EPOCH).total_seconds() / 86400.0


class _NowTime(object):
    @property
    def mjd(self):
        return _mjd()

    @property
    def iso(self):
        return _NOW[0].strftime('%Y-%m-%d %H:%M:%S')

    @property
    def lv(self):
        # LabVIEW epoch seconds, matching util.Time.lv for the synthetic clock.
        return _mjd() * 86400.0 + 2082844800.0

    @classmethod
    def now(cls):
        return cls()


class _Util(object):
    Time = _NowTime

    @staticmethod
    def ant_str2list(spec):
        result = []
        for token in spec.split():
            values = token[3:].split('-')
            if len(values) == 1:
                result.append(int(values[0]) - 1)
            else:
                result.extend(range(int(values[0]) - 1, int(values[1])))
        return result


class _StateFrame(object):
    @staticmethod
    def extract(data, key):
        return data[key]


def _macro_commands(cmds):
    if cmds and cmds[0].upper() == 'PHASECAL':
        return ['$SCAN-STOP', '$PA-EXIT', '$WAIT 2',
                '$SUBARRAY default.antlist phasecal',
                '$WAIT 2', '$SCAN-START']
    return ['$SCAN-STOP', '$WAIT 2', '$MK_TABLES sun_tab Sun',
            '$WAIT 5', '$SCAN-START']


def _resolve_subarray_args(args):
    if len(args) >= 2 and args[0].lower().endswith('.antlist'):
        return {
            'phasecal': 'ant1 ant3-6 ant8-11 ant13 ant15 ant16',
        }.get(args[1], '')
    return ' '.join(args)


class _Var(object):
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _Rows(object):
    def __init__(self, rows):
        self.rows = list(rows)
        self.colors = {}

    def get(self, index):
        return self.rows[index]

    def size(self):
        return len(self.rows)

    def delete(self, index, end=None):
        if end is None:
            del self.rows[index]
        else:
            del self.rows[index:]

    def insert(self, index, value):
        self.rows.insert(index, value)

    def itemcget(self, index, option):
        return self.colors.get(index, 'white')

    def itemconfig(self, index, **kwargs):
        if 'background' in kwargs:
            self.colors[index] = kwargs['background']

    def see(self, index):
        pass

    def configure(self, **kwargs):
        pass

    def selection_clear(self, start, end):
        pass

    def selection_set(self, index):
        pass

    def curselection(self):
        return ()


class _Widget(object):
    def __init__(self):
        self.options = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)


def _load_scheduler_methods():
    source_path = os.path.join(os.path.dirname(__file__), 'schedule.py')
    with open(source_path) as source_file:
        source = source_file.read()
    if sys.version_info[0] >= 3:
        source = str(RefactoringTool(
            get_fixers_from_package('lib2to3.fixes')).refactor_string(
                source, source_path))
    tree = ast.parse(source)
    app = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'App')
    inc_time = next(node for node in app.body
                    if isinstance(node, ast.FunctionDef) and
                    node.name == 'inc_time')
    go_block = next(node for node in inc_time.body
                    if isinstance(node, ast.If) and
                    isinstance(node.test, ast.Compare) and
                    isinstance(node.test.left, ast.Attribute) and
                    node.test.left.attr == 'Toggle')
    wrapper = ast.parse(
        'def _run_inc_go_block(self, data, msg):\n    pass\n').body[0]
    wrapper.body = [copy.deepcopy(go_block)]
    wanted = {
        '_active_skip_indices', '_advance_running_line',
        '_ant16_in_spec', '_auto_wind_pair_key', '_phasecal_uses_ant16',
        '_ensure_current_line_valid', '_is_midday_phacal',
        '_mark_line_skipped', '_maybe_auto_skip_current_pair',
        '_next_active_index', '_phasecal_pair_index',
        '_record_skip_phacal_line', '_refresh_auto_wind_skip_indices',
        '_restore_auto_wind_skip_state', '_schedule_lines',
        '_update_skip_phacal_indices', '_apply_skip_phacal_styling',
        'on_skip_phacal_toggle',
        '_update_auto_wind_controller_health',
        '_auto_wind_controller_status',
        '_auto_wind_recovery_candidate', '_latch_auto_wind_pair',
        '_log_auto_wind_recovery', '_maybe_recover_active_pair',
        '_recovery_override', 'execute_cmds',
        '_start_due_line', '_activate_auto_wind_skip',
        'toggle_state',
    }
    app.body = [node for node in app.body
                if isinstance(node, ast.FunctionDef)
                and node.name in wanted]
    app.body.append(wrapper)
    ast.fix_missing_locations(app)
    namespace = {
        'END': 1000000000,
        'NORMAL': 'normal',
        'DISABLED': 'disabled',
        'AUTO_WIND_SKIP_STALE_SECONDS': 300.0,
        'AUTO_WIND_ACC_STALE_SECONDS': 30.0,
        'AUTO_WIND_CONTROLLER_GRACE_SECONDS': 5.0,
        'TclError': Exception,
        'mjd': _mjd,
        'util': _Util,
        'Time': _NowTime,
        'stateframe': _StateFrame,
        'evaluate_controller_clocks': evaluate_controller_clocks,
        'numpy': numpy,
        'json': json,
        'sys': sys,
        'sf_dict': {},
        'sh_dict': {},
        'ANT13_FEM_BUILTIN_COMMANDS': {},
        '_read_macro_commands': _macro_commands,
        'get_antlist': lambda key, filename: {
            'phasecal': 'ant1 ant3-6 ant8-11 ant13 ant15 ant16',
        }.get(key, ''),
        'resolve_subarray_args': _resolve_subarray_args,
    }
    if sys.version_info[0] >= 3:
        module = ast.Module(body=[app], type_ignores=[])
    else:
        module = ast.Module(body=[app])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['App']


SchedulerMethods = _load_scheduler_methods()


def _line(day, clock, command):
    return '%s %s %s' % (day, clock, command)


def _schedule_lines():
    return [
        _line('2026-09-06', '15:00:00', 'SUN'),
        _line('2026-09-06', '15:10:00', 'ACQUIRE 0319+415'),
        _line('2026-09-06', '15:14:00', 'PHASECAL 0319+415 pcal_hi'),
        _line('2026-09-06', '15:30:00', 'SUN'),
        _line('2026-09-06', '16:30:00', 'SOLPNTCAL solar.fsq'),
    ]


class _SchedulerHarness(SchedulerMethods):
    def __init__(self, rows=None):
        self.L = _Rows(rows or _schedule_lines())
        self.status = _Rows(['Waiting...'] * self.L.size())
        self.L2 = _Rows([])
        self.lastline = self.L.size()
        self.curline = 0
        self.auto_wind_skip = _Var(True)
        self.skip_phacal = _Var(False)
        self.skip_phacal_indices = set()
        self.auto_wind_skip_indices = set()
        self.auto_wind_decisions = {}
        self.auto_wind_recovery = {}
        self.auto_wind_controller_health = {
            'last_root_timestamp': 1.0,
            'last_healthy_mjd': _mjd(),
            'last_check_mjd': _mjd(),
            'last_result': {
                'available': True,
                'reason': 'Ant16 controller clocks fresh',
                'crio_age_seconds': 0.1,
                'system_age_seconds': 0.1,
                'acc_age_seconds': 0.1,
            },
            'seen_frame': True,
        }
        self.waitmode = False
        self.nextctlline = 0
        self.wait = 0
        self.PAthread = None
        self.Toggle = 0
        self.ant13_fem_ready_deadline = None
        self.B2 = _Widget()
        self.downbutton = _Widget()
        self.upbutton = _Widget()
        self.Insert = _Widget()
        self.ClearBtn = _Widget()
        self.TodayBtn = _Widget()
        self.executed_skip_phacal_indices = set()
        self.executed_skip_phacal_lines = []
        self.wlimit = 17
        self.commands = []
        self.decisions = []
        controller = {
            'cRIOClockms': 'crio_clock_ms',
            'SystemClockMJDay': 'system_clock_mjday',
            'SystemClockms': 'system_clock_ms',
        }
        windscram = {
            'State': 'scram_state',
            'CommErr': 'scram_comm_err',
        }
        antenna = [{'Controller': controller,
                    'Frontend': {'WindScram': windscram}}
                   for _ in range(16)]
        self.accini = {'sf': {'Timestamp': 'timestamp',
                              'Antenna': antenna}}

        self._phasecal_uses_ant16 = lambda index: True
        self._auto_wind_telemetry = lambda now, data, msg: {
            'skip': self.wind_skip,
            'reason': 'wind at or above limit' if self.wind_skip else
                      'wind within limit',
            'wind_mph': 18 if self.wind_skip else 10,
            'wind_limit_mph': 17,
            'sample_time': '2026-09-06 15:09:30',
            'sample_age_seconds': 30,
            'scram_state': 1 if self.wind_skip else 0,
            'scram_comm_err': 0,
            'wind_confirmed': self.wind_skip,
        }
        self._log_auto_wind_decision = lambda *args: self.decisions.append(args)

    def controller_frame(self, age_seconds=1.0, root_offset_seconds=0.0,
                         clock_offset_ms=0):
        frame_mjd = _mjd() - age_seconds / 86400.0
        day = int(frame_mjd)
        clock_ms = int(round((frame_mjd - day) * 86400000.0))
        return {
            'timestamp': _NowTime.now().lv - age_seconds -
                         root_offset_seconds,
            'crio_clock_ms': clock_ms + clock_offset_ms,
            'system_clock_mjday': day,
            'system_clock_ms': clock_ms,
            'scram_state': 0,
            'scram_comm_err': 0,
        }

    def update_status(self):
        pass

    def execute_cmds(self):
        self.commands.append(self.curline)

    def execute_ctlline(self, line, mjd1=None, mjd2=None):
        self.commands.append((line, mjd1, mjd2))


class _MacroSchedulerHarness(_SchedulerHarness):
    execute_cmds = SchedulerMethods.execute_cmds

    def execute_ctlline(self, line, mjd1=None, mjd2=None):
        self.commands.append((line, mjd1, mjd2))
        if line.split()[0].upper() == '$WAIT':
            self.waitmode = True
            self.wait = int(line.split()[1])


class _AliveThread(object):
    def is_alive(self):
        return True


class ScheduleWindProgressionTests(unittest.TestCase):

    def setUp(self):
        _NOW[0] = datetime(2026, 9, 6, 15, 10, 0)

    def test_due_windy_acquire_latches_pair_and_keeps_sun_running(self):
        scheduler = _SchedulerHarness()
        scheduler.wind_skip = True
        scheduler.curline = 0
        scheduler.status.delete(0)
        scheduler.status.insert(0, 'Running...')

        scheduler._advance_running_line(_mjd(), None, 'No Error')

        self.assertEqual(scheduler.curline, 0)
        self.assertEqual(scheduler.status.get(0), 'Running...')
        self.assertEqual(scheduler.status.get(1), 'Skipped')
        self.assertEqual(scheduler.status.get(2), 'Skipped')
        self.assertEqual(scheduler.commands, [])

        # The next SUN is still in the future, so later ticks must not look
        # ahead and stop the current SUN merely because the pair was skipped.
        _NOW[0] += timedelta(minutes=10)
        scheduler._advance_running_line(_mjd(), None, 'No Error')
        self.assertEqual(scheduler.curline, 0)
        self.assertEqual(scheduler.commands, [])

        _NOW[0] += timedelta(minutes=10)
        scheduler._advance_running_line(_mjd(), None, 'No Error')
        self.assertEqual(scheduler.curline, 3)
        self.assertEqual(scheduler.status.get(0), 'Done')
        self.assertEqual(scheduler.commands, [3])

    def test_due_calm_acquire_executes_normally(self):
        scheduler = _SchedulerHarness()
        scheduler.wind_skip = False
        scheduler.curline = 0
        scheduler.status.delete(0)
        scheduler.status.insert(0, 'Running...')

        scheduler._advance_running_line(_mjd(), None, 'No Error')

        self.assertEqual(scheduler.curline, 1)
        self.assertEqual(scheduler.status.get(0), 'Done')
        self.assertEqual(scheduler.status.get(1), 'Running...')
        self.assertEqual(scheduler.commands, [1])
        self.assertEqual(scheduler.auto_wind_skip_indices, set())

    def test_default_antlist_subarray_detects_ant16(self):
        scheduler = _SchedulerHarness()
        self.assertTrue(scheduler._phasecal_uses_ant16(2))
        self.assertIsNone(scheduler._ant16_in_spec('not-an-antenna'))

    def test_calm_unknown_weather_executes_with_healthy_controller(self):
        scheduler = _SchedulerHarness()
        scheduler.curline = 1
        scheduler.status.delete(1)
        scheduler.status.insert(1, 'Waiting...')
        scheduler._auto_wind_telemetry = lambda now, data, msg: {
            'skip': False,
            'reason': 'weather unavailable',
            'weather_available': False,
            'scram_reliable': True,
            'wind_confirmed': False,
        }

        self.assertTrue(scheduler._start_due_line(
            _NowTime.now(), None, 'No Error'))
        self.assertEqual(scheduler.curline, 1)
        self.assertEqual(scheduler.commands, [1])
        self.assertFalse(scheduler.auto_wind_skip_indices)

    def test_startup_unknown_controller_is_a_distinct_conservative_skip(self):
        scheduler = _SchedulerHarness()
        scheduler.auto_wind_controller_health = {
            'last_root_timestamp': None,
            'last_healthy_mjd': None,
            'last_check_mjd': None,
            'last_result': None,
            'seen_frame': False,
        }
        scheduler.wind_skip = False
        scheduler.curline = 1
        scheduler.status.delete(1)
        scheduler.status.insert(1, 'Waiting...')

        self.assertTrue(scheduler._start_due_line(
            _NowTime.now(), None, 'No Error'))
        key = scheduler._auto_wind_pair_key(2)
        self.assertEqual(scheduler.auto_wind_decisions[key]['reason'],
                         'Ant16 controller availability unknown')
        self.assertEqual(scheduler.auto_wind_decisions[key]
                         ['controller_health_state'], 'unknown')

    def test_offline_controller_skips_calm_pair(self):
        scheduler = _SchedulerHarness()
        scheduler.auto_wind_controller_health['last_healthy_mjd'] = (
            _mjd() - 10.0 / 86400.0)
        scheduler.auto_wind_controller_health['last_result'] = {
            'available': False,
            'reason': 'Ant16 controller clock stale',
        }
        scheduler.curline = 1
        scheduler.status.delete(1)
        scheduler.status.insert(1, 'Waiting...')
        scheduler.wind_skip = False

        self.assertTrue(scheduler._start_due_line(
            _NowTime.now(), None, 'No Error'))
        key = scheduler._auto_wind_pair_key(2)
        self.assertEqual(scheduler.auto_wind_decisions[key]['reason'],
                         'Ant16 controller unavailable')
        self.assertEqual(scheduler.auto_wind_decisions[key]
                         ['controller_health_state'], 'offline')

    def test_controller_health_grace_does_not_refresh_repeated_frame(self):
        scheduler = _SchedulerHarness()
        scheduler.auto_wind_controller_health = {
            'last_root_timestamp': None,
            'last_healthy_mjd': None,
            'last_check_mjd': None,
            'last_result': None,
            'seen_frame': False,
        }
        frame = scheduler.controller_frame(age_seconds=1.0)
        scheduler._update_auto_wind_controller_health(
            _NowTime.now(), frame, 'No Error')
        healthy_mjd = scheduler.auto_wind_controller_health['last_healthy_mjd']
        self.assertTrue(scheduler.auto_wind_controller_health['last_result']
                        ['available'])

        _NOW[0] += timedelta(seconds=2)
        scheduler._update_auto_wind_controller_health(
            _NowTime.now(), frame, 'No Error')
        status = scheduler._auto_wind_controller_status(_NowTime.now())
        self.assertEqual(status['health_state'], 'grace')
        self.assertEqual(scheduler.auto_wind_controller_health
                         ['last_healthy_mjd'], healthy_mjd)

        _NOW[0] += timedelta(seconds=4)
        status = scheduler._auto_wind_controller_status(_NowTime.now())
        self.assertEqual(status['health_state'], 'offline')
        self.assertEqual(scheduler.auto_wind_controller_health
                         ['last_healthy_mjd'], healthy_mjd)

    def test_old_advancing_frame_does_not_extend_controller_grace(self):
        scheduler = _SchedulerHarness()
        scheduler.auto_wind_controller_health = {
            'last_root_timestamp': None,
            'last_healthy_mjd': None,
            'last_check_mjd': None,
            'last_result': None,
            'seen_frame': False,
        }
        old_frame = scheduler.controller_frame(age_seconds=20.0)
        scheduler._update_auto_wind_controller_health(
            _NowTime.now(), old_frame, 'No Error')
        self.assertTrue(scheduler.auto_wind_controller_health['last_result']
                        ['available'])
        status = scheduler._auto_wind_controller_status(_NowTime.now())
        self.assertEqual(status['health_state'], 'offline')

    def test_nonfinite_controller_clock_is_not_healthy(self):
        scheduler = _SchedulerHarness()
        scheduler.auto_wind_controller_health = {
            'last_root_timestamp': None,
            'last_healthy_mjd': None,
            'last_check_mjd': None,
            'last_result': None,
            'seen_frame': False,
        }
        frame = scheduler.controller_frame()
        frame['crio_clock_ms'] = float('nan')
        scheduler._update_auto_wind_controller_health(
            _NowTime.now(), frame, 'No Error')
        self.assertFalse(scheduler.auto_wind_controller_health['last_result']
                         ['available'])
        self.assertIsNone(scheduler.auto_wind_controller_health
                          ['last_healthy_mjd'])

    def test_decreasing_or_future_frame_does_not_replace_last_valid_frame(self):
        scheduler = _SchedulerHarness()
        scheduler.auto_wind_controller_health = {
            'last_root_timestamp': None,
            'last_healthy_mjd': None,
            'last_check_mjd': None,
            'last_result': None,
            'seen_frame': False,
        }
        first = scheduler.controller_frame(age_seconds=1.0)
        scheduler._update_auto_wind_controller_health(
            _NowTime.now(), first, 'No Error')
        accepted_root = scheduler.auto_wind_controller_health[
            'last_root_timestamp']
        decreasing = dict(first)
        decreasing['timestamp'] -= 2.0
        scheduler._update_auto_wind_controller_health(
            _NowTime.now(), decreasing, 'No Error')
        self.assertEqual(scheduler.auto_wind_controller_health
                         ['last_root_timestamp'], accepted_root)
        future = dict(first)
        future['timestamp'] = _NowTime.now().lv + 1.0
        scheduler._update_auto_wind_controller_health(
            _NowTime.now(), future, 'No Error')
        self.assertEqual(scheduler.auto_wind_controller_health
                         ['last_root_timestamp'], accepted_root)

    def test_manual_skip_toggle_remains_independent_of_auto_latch(self):
        scheduler = _SchedulerHarness()
        scheduler.auto_wind_skip.value = False
        scheduler.skip_phacal.value = True
        scheduler._update_skip_phacal_indices()
        scheduler.curline = 1
        scheduler._ensure_current_line_valid()
        self.assertEqual(scheduler.curline, 3)
        self.assertEqual(scheduler.status.get(1), 'Skipped')
        self.assertEqual(scheduler.status.get(2), 'Skipped')
        self.assertEqual(scheduler.auto_wind_skip_indices, set())

        scheduler.skip_phacal.value = False
        scheduler.on_skip_phacal_toggle()
        self.assertEqual(scheduler.auto_wind_skip_indices, set())

    def test_start_and_resume_skip_latched_rows(self):
        scheduler = _SchedulerHarness()
        scheduler.wind_skip = True
        scheduler.curline = 1
        scheduler._start_due_line(_mjd(), None, 'No Error')
        self.assertEqual(scheduler.curline, 1)
        self.assertEqual(scheduler.status.get(1), 'Skipped')
        self.assertEqual(scheduler.status.get(2), 'Skipped')
        self.assertEqual(scheduler.commands, [])

        # A restart/resume that lands on PHASECAL must consume the restored
        # latch before trying to start that row.
        scheduler.curline = 2
        scheduler._ensure_current_line_valid()
        self.assertEqual(scheduler.curline, 3)
        self.assertEqual(scheduler.commands, [])

        _NOW[0] += timedelta(minutes=20)
        scheduler._start_due_line(_mjd(), None, 'No Error')
        self.assertEqual(scheduler.curline, 3)
        self.assertEqual(scheduler.status.get(3), 'Running...')
        self.assertEqual(scheduler.commands, [3])

    def test_confirmed_wind_recovers_active_acquire_to_next_sun(self):
        scheduler = _SchedulerHarness()
        scheduler.wind_skip = True
        scheduler.curline = 1
        scheduler.status.delete(1)
        scheduler.status.insert(1, 'Running...')

        self.assertTrue(scheduler._maybe_recover_active_pair(None, 'No Error'))
        self.assertEqual(scheduler.curline, 3)
        self.assertEqual(scheduler.status.get(1), 'Skipped')
        self.assertEqual(scheduler.status.get(2), 'Skipped')
        self.assertEqual(scheduler.status.get(3), 'Running...')
        self.assertEqual(scheduler.commands, [3])
        self.assertEqual(len(scheduler.auto_wind_recovery), 1)

        # The original 15:30 row remains unchanged, so the active recovered
        # SUN is not restarted at its nominal time and later progress remains
        # tied to the following schedule entry.
        _NOW[0] = datetime(2026, 9, 6, 15, 30, 0)
        self.assertFalse(scheduler._maybe_recover_active_pair(None, 'No Error'))
        scheduler._advance_running_line(_mjd(), None, 'No Error')
        self.assertEqual(scheduler.curline, 3)
        _NOW[0] = datetime(2026, 9, 6, 16, 30, 0)
        scheduler._advance_running_line(_mjd(), None, 'No Error')
        self.assertEqual(scheduler.curline, 4)
        self.assertEqual(scheduler.auto_wind_recovery, {})

    def test_confirmed_wind_recovers_active_phasecal(self):
        scheduler = _SchedulerHarness()
        scheduler.wind_skip = True
        scheduler.curline = 2
        scheduler.status.delete(2)
        scheduler.status.insert(2, 'Running...')

        self.assertTrue(scheduler._maybe_recover_active_pair(None, 'No Error'))
        self.assertEqual(scheduler.curline, 3)
        self.assertEqual(scheduler.commands, [3])

    def test_confirmed_wind_does_not_interrupt_active_sun(self):
        scheduler = _SchedulerHarness()
        scheduler.wind_skip = True
        scheduler.curline = 0
        scheduler.status.delete(0)
        scheduler.status.insert(0, 'Running...')

        self.assertFalse(scheduler._maybe_recover_active_pair(None,
                                                               'No Error'))
        self.assertEqual(scheduler.curline, 0)
        self.assertEqual(scheduler.commands, [])
        self.assertEqual(scheduler.auto_wind_recovery, {})

    def test_recovery_exits_active_pa_tracking(self):
        scheduler = _SchedulerHarness()
        scheduler.wind_skip = True
        scheduler.PAthread = _AliveThread()
        scheduler.curline = 2
        scheduler.status.delete(2)
        scheduler.status.insert(2, 'Running...')

        self.assertTrue(scheduler._maybe_recover_active_pair(None, 'No Error'))
        self.assertEqual(scheduler.commands[0][0], '$PA-EXIT')

    def test_recovery_requires_positive_fresh_wind_and_solar_following_entry(self):
        scheduler = _SchedulerHarness()
        scheduler.curline = 1
        scheduler.status.delete(1)
        scheduler.status.insert(1, 'Running...')
        scheduler.wind_skip = False
        self.assertFalse(scheduler._maybe_recover_active_pair(None, 'No Error'))
        self.assertEqual(scheduler.curline, 1)

        scheduler.wind_skip = True
        scheduler.auto_wind_skip.value = False
        self.assertFalse(scheduler._maybe_recover_active_pair(None, 'No Error'))
        self.assertEqual(scheduler.curline, 1)

        scheduler.auto_wind_skip.value = True
        scheduler._phasecal_uses_ant16 = lambda index: None
        self.assertFalse(scheduler._maybe_recover_active_pair(None, 'No Error'))
        self.assertEqual(scheduler.curline, 1)

        scheduler._phasecal_uses_ant16 = lambda index: True
        scheduler._auto_wind_telemetry = lambda now, data, msg: {
            'skip': True, 'wind_confirmed': False,
        }
        self.assertFalse(scheduler._maybe_recover_active_pair(None, 'No Error'))
        self.assertEqual(scheduler.curline, 1)

        no_following = _SchedulerHarness(_schedule_lines()[:4])
        no_following.curline = 1
        no_following.status.delete(1)
        no_following.status.insert(1, 'Running...')
        no_following.wind_skip = True
        self.assertFalse(no_following._maybe_recover_active_pair(None, 'No Error'))

        ant3_rows = _schedule_lines()[:3] + [
            _line('2026-09-06', '15:30:00', 'SUN_ANT3'),
            _line('2026-09-06', '16:30:00', 'SOLPNTCAL solar.fsq')]
        ant3 = _SchedulerHarness(ant3_rows)
        ant3.curline = 1
        ant3.status.delete(1)
        ant3.status.insert(1, 'Running...')
        ant3.wind_skip = True
        self.assertFalse(ant3._maybe_recover_active_pair(None, 'No Error'))

        sun_no_ant3_rows = _schedule_lines()[:3] + [
            _line('2026-09-06', '15:30:00', 'SUN_NO_ANT3'),
            _line('2026-09-06', '16:30:00', 'SOLPNTCAL solar.fsq')]
        sun_no_ant3 = _SchedulerHarness(sun_no_ant3_rows)
        sun_no_ant3.curline = 1
        sun_no_ant3.status.delete(1)
        sun_no_ant3.status.insert(1, 'Running...')
        sun_no_ant3.wind_skip = True
        self.assertTrue(sun_no_ant3._maybe_recover_active_pair(
            None, 'No Error'))
        self.assertEqual(sun_no_ant3.curline, 3)

    def test_real_inc_time_go_branch_aborts_waiting_phasecal_before_continuation(self):
        scheduler = _MacroSchedulerHarness()
        scheduler.wind_skip = True
        scheduler.curline = 2
        scheduler.status.delete(2)
        scheduler.status.insert(2, 'Running...')
        scheduler.waitmode = True
        scheduler.wait = 1
        scheduler.nextctlline = 4

        # Execute the production GO branch, rather than calling the recovery
        # helper directly.  The old phasecal continuation is deliberately
        # pending so this catches recovery being checked after $WAIT handling.
        scheduler._run_inc_go_block(None, 'No Error')
        self.assertEqual(scheduler.curline, 3)
        self.assertFalse(scheduler.nextctlline == 4)
        self.assertEqual(scheduler.commands[0][0], '$SCAN-STOP')
        self.assertTrue(any(command[0] == '$WAIT 2'
                            for command in scheduler.commands))
        self.assertFalse(any('phasecal' in command[0].lower()
                             for command in scheduler.commands))
        recovery_mjd = scheduler.commands[0][1]
        self.assertAlmostEqual(recovery_mjd, _mjd(), places=8)

        # A later GO tick resumes the SUN macro at its real $MK_TABLES line,
        # retaining the recovery timestamp through the wait boundary.
        scheduler.wait = 1
        scheduler._run_inc_go_block(None, 'No Error')
        mk_tables = [command for command in scheduler.commands
                     if command[0].startswith('$MK_TABLES')]
        self.assertEqual(len(mk_tables), 1)
        self.assertEqual(mk_tables[0][1], recovery_mjd)
        self.assertEqual(scheduler.curline, 3)

    def test_toggle_stop_go_restarts_completed_recovery_sun_early(self):
        scheduler = _MacroSchedulerHarness()
        scheduler.auto_wind_skip_indices = {1, 2}
        scheduler.auto_wind_recovery[scheduler.L.get(3)] = {
            'mjd1': 999.25,
        }
        scheduler.curline = 3
        scheduler.status.delete(3)
        scheduler.status.insert(3, 'Done')
        scheduler.Toggle = 0

        # The recovered SUN has completed, but its nominal schedule time is
        # still ahead.  Stop and Go must retain the early-start override.
        _NOW[0] = datetime(2026, 9, 6, 15, 20, 0)
        scheduler.toggle_state()
        self.assertEqual(scheduler.Toggle, 1)
        scheduler.toggle_state()
        self.assertEqual(scheduler.Toggle, 0)
        self.assertEqual(scheduler.curline, 3)
        self.assertEqual(scheduler.status.get(3), 'Started...')
        self.assertEqual(scheduler.auto_wind_recovery[scheduler.L.get(3)]['mjd1'],
                         999.25)

        scheduler._run_inc_go_block(None, 'No Error')
        self.assertEqual(scheduler.commands[0][0], '$SCAN-STOP')
        self.assertEqual(scheduler.commands[0][1], 999.25)

    def test_manual_stop_tick_does_not_recover_active_phasecal(self):
        scheduler = _MacroSchedulerHarness()
        scheduler.wind_skip = True
        scheduler.Toggle = 1
        scheduler.curline = 2
        scheduler.status.delete(2)
        scheduler.status.insert(2, 'Running...')
        scheduler.waitmode = True
        scheduler.wait = 1
        scheduler.nextctlline = 4

        # The extracted production GO block includes the Toggle guard.  A
        # stopped schedule must leave the active calibration untouched.
        scheduler._run_inc_go_block(None, 'No Error')
        self.assertEqual(scheduler.curline, 2)
        self.assertEqual(scheduler.wait, 1)
        self.assertEqual(scheduler.nextctlline, 4)
        self.assertEqual(scheduler.auto_wind_recovery, {})
        self.assertEqual(scheduler.commands, [])

    def test_recovery_override_survives_macro_waits_and_line_advance(self):
        scheduler = _MacroSchedulerHarness()
        scheduler.curline = 3
        scheduler.auto_wind_recovery[scheduler.L.get(3)] = {
            'mjd1': 999.25,
        }

        scheduler.execute_cmds()
        self.assertTrue(scheduler.waitmode)
        self.assertEqual(scheduler.commands[0][0], '$SCAN-STOP')
        self.assertEqual(scheduler.commands[0][1], 999.25)
        self.assertIn(scheduler.L.get(3), scheduler.auto_wind_recovery)

        scheduler.execute_cmds()
        self.assertEqual(scheduler.commands[2][0], '$MK_TABLES sun_tab Sun')
        self.assertEqual(scheduler.commands[2][1], 999.25)
        self.assertIn(scheduler.L.get(3), scheduler.auto_wind_recovery)

        scheduler.execute_cmds()
        self.assertEqual(scheduler.commands[4][0], '$SCAN-START')
        self.assertIn(scheduler.L.get(3), scheduler.auto_wind_recovery)

        # Stop/Go while the recovered SUN is still active must restart that
        # same SUN early, rather than waiting for its original row time.
        scheduler.auto_wind_skip_indices = {1, 2}
        scheduler.curline = 2
        scheduler.status.delete(2)
        scheduler.status.insert(2, 'Skipped')
        scheduler.status.delete(3)
        scheduler.status.insert(3, '')
        scheduler._ensure_current_line_valid()
        self.assertEqual(scheduler.curline, 3)
        self.assertEqual(scheduler.status.get(3), 'Started...')
        scheduler.waitmode = False
        scheduler.execute_cmds()
        self.assertEqual(scheduler.commands[5][0], '$SCAN-STOP')
        self.assertEqual(scheduler.commands[5][1], 999.25)
        self.assertIn(scheduler.L.get(3), scheduler.auto_wind_recovery)

    def test_restore_rebuilds_latch_from_exact_pair_rows(self):
        scheduler = _SchedulerHarness()
        scheduler.executed_skip_phacal_lines = scheduler.L.rows[1:3]
        scheduler.auto_wind_skip_indices = set()
        scheduler.auto_wind_decisions = {}

        scheduler._restore_auto_wind_skip_state()

        self.assertEqual(scheduler.auto_wind_skip_indices, {1, 2})
        self.assertEqual(len(scheduler.auto_wind_decisions), 1)

    def test_resume_with_manual_skip_waits_for_blank_future_sun(self):
        scheduler = _SchedulerHarness()
        scheduler.wind_skip = True
        scheduler.skip_phacal.value = True
        scheduler.skip_phacal_indices = {1, 2}
        scheduler.auto_wind_skip_indices = {1, 2}
        scheduler.curline = 2
        scheduler.status.rows = ['Skipped', 'Skipped', 'Started...', '']
        scheduler._ensure_current_line_valid()
        self.assertEqual(scheduler.curline, 3)
        self.assertEqual(scheduler.status.get(3), 'Waiting...')
        scheduler._start_due_line(_mjd(), None, 'No Error')
        self.assertEqual(scheduler.commands, [])
        _NOW[0] += timedelta(minutes=20)
        scheduler._start_due_line(_mjd(), None, 'No Error')
        self.assertEqual(scheduler.commands, [3])

    def test_edit_save_reload_preserves_remapped_pair(self):
        original = _schedule_lines()
        scheduler = _SchedulerHarness(original)
        key = (original[1], original[2])
        scheduler.auto_wind_decisions[key] = {'skip': True}
        scheduler.auto_wind_skip_indices = {1, 2}
        scheduler.executed_skip_phacal_lines = original[1:3]

        edited = [
            original[0],
            _line('2026-09-06', '14:00:00', 'ACQUIRE 0000+000'),
            _line('2026-09-06', '14:04:00', 'PHASECAL 0000+000 pcal_hi'),
        ] + original[1:]
        scheduler.L = _Rows(edited)
        scheduler.lastline = len(edited)
        scheduler._refresh_auto_wind_skip_indices()

        state_path = os.path.join(tempfile.mkdtemp(), 'skip.json')
        try:
            write_executed_lines(
                state_path, edited, scheduler.executed_skip_phacal_lines)
            fresh = _SchedulerHarness(edited)
            fresh.executed_skip_phacal_lines = load_executed_lines(
                state_path, edited)
            fresh._restore_auto_wind_skip_state()
            self.assertEqual(fresh.auto_wind_skip_indices, {3, 4})
        finally:
            os.unlink(state_path)
            os.rmdir(os.path.dirname(state_path))

    def test_pair_identity_survives_inserting_or_removing_earlier_pair(self):
        original = _schedule_lines()
        scheduler = _SchedulerHarness(original)
        key = (original[1], original[2])
        scheduler.auto_wind_decisions[key] = {'skip': True}
        scheduler.auto_wind_skip_indices = {1, 2}

        inserted = [
            original[0],
            _line('2026-09-06', '14:00:00', 'ACQUIRE 0000+000'),
            _line('2026-09-06', '14:04:00', 'PHASECAL 0000+000 pcal_hi'),
        ] + original[1:]
        scheduler.L = _Rows(inserted)
        scheduler.lastline = len(inserted)
        scheduler._refresh_auto_wind_skip_indices()
        self.assertEqual(scheduler.auto_wind_skip_indices, {3, 4})

        scheduler.L = _Rows(original)
        scheduler.lastline = len(original)
        scheduler._refresh_auto_wind_skip_indices()
        self.assertEqual(scheduler.auto_wind_skip_indices, {1, 2})

    def test_sun_boundaries_exclude_refcals(self):
        dawn = [
            _line('2026-09-05', '15:30:00', 'SUN'),
            _line('2026-09-05', '16:00:00', 'STOW_ANT3'),
            _line('2026-09-06', '13:00:00', 'ACQUIRE 0319+415'),
            _line('2026-09-06', '13:04:00', 'PHASECAL 0319+415 pcal_hi'),
            _line('2026-09-06', '13:20:00', 'SUN'),
        ]
        simple_dawn = [
            _line('2026-09-06', '13:00:00', 'ACQUIRE 0319+415'),
            _line('2026-09-06', '13:04:00', 'PHASECAL 0319+415 pcal_hi'),
            _line('2026-09-06', '13:20:00', 'SUN'),
        ]
        dusk = [
            _line('2026-09-06', '23:00:00', 'SUN'),
            _line('2026-09-06', '23:10:00', 'ACQUIRE 0319+415'),
            _line('2026-09-06', '23:14:00', 'PHASECAL 0319+415 pcal_hi'),
            _line('2026-09-07', '00:00:00', 'STOW_ANT3'),
        ]
        no_next_sun = dusk[:3]
        for rows in (dawn, simple_dawn, dusk, no_next_sun):
            scheduler = _SchedulerHarness(rows)
            phase_idx = 3 if rows is dawn else 1 if rows is simple_dawn else 2
            self.assertFalse(scheduler._is_midday_phacal(phase_idx))

        # The active-calibration recovery path uses the same strict boundaries
        # and must not turn a morning or evening refcal into an early SUN.
        for rows, acq_idx in ((dawn, 2), (simple_dawn, 0), (dusk, 1),
                              (no_next_sun, 1)):
            scheduler = _SchedulerHarness(rows)
            scheduler.curline = acq_idx
            scheduler.status.delete(acq_idx)
            scheduler.status.insert(acq_idx, 'Running...')
            scheduler.wind_skip = True
            self.assertFalse(scheduler._maybe_recover_active_pair(
                None, 'No Error'))


if __name__ == '__main__':
    unittest.main()
