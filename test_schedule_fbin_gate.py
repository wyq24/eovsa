"""Isolated tests for the scheduler's phase-three fBin gate.

``schedule.py`` is a Python 2 Tk application with live-control imports, so
these tests extract only the fBin conditional after applying the same 2to3
translation used by the scheduler tests.  No scheduler module or hardware
endpoint is imported.
"""

import ast
import copy
import os
import sys
import unittest
from lib2to3.refactor import RefactoringTool, get_fixers_from_package


_SOURCE_PATH = os.path.join(os.path.dirname(__file__), 'schedule.py')
_REFACTOR = RefactoringTool(get_fixers_from_package('lib2to3.fixes'))


def _tree():
    with open(_SOURCE_PATH) as source_file:
        source = source_file.read()
    if sys.version_info[0] >= 3:
        source = str(_REFACTOR.refactor_string(source, _SOURCE_PATH))
    return ast.parse(source)


def _string_constants(node):
    return [item.value for item in ast.walk(node)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)]


def _fbin_if():
    tree = _tree()
    app = next(item for item in tree.body
               if isinstance(item, ast.ClassDef) and item.name == 'App')
    inc_time = next(item for item in app.body
                    if isinstance(item, ast.FunctionDef) and item.name == 'inc_time')
    for index, item in enumerate(inc_time.body):
        if isinstance(item, ast.If) and any('fBin' in value for value in _string_constants(item)):
            return inc_time, index, item
    raise AssertionError('could not find the scheduler fBin conditional')


def _flag_value(environment_value):
    old_value = os.environ.get('EOVSA_DISABLE_FBIN')
    try:
        if environment_value is None:
            os.environ.pop('EOVSA_DISABLE_FBIN', None)
        else:
            os.environ['EOVSA_DISABLE_FBIN'] = environment_value
        assignment = next(item for item in _tree().body
                          if isinstance(item, ast.Assign) and
                          any(getattr(target, 'id', None) == 'DISABLE_FBIN'
                              for target in item.targets))
        module = ast.Module(body=[copy.deepcopy(assignment)], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {'os': os}
        exec(compile(module, _SOURCE_PATH, 'exec'), namespace)
        return namespace['DISABLE_FBIN']
    finally:
        if old_value is None:
            os.environ.pop('EOVSA_DISABLE_FBIN', None)
        else:
            os.environ['EOVSA_DISABLE_FBIN'] = old_value


class _Connection(object):
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


class _Cursor(object):
    def __init__(self):
        self.calls = []

    def execute(self, *args):
        self.calls.append(args)


class _StateframeDef(object):
    class pyodbc(object):
        @staticmethod
        def Binary(value):
            return value

    @staticmethod
    def transmogrify(data, _definition):
        return b'transformed:' + data


class _Scheduler(object):
    def __init__(self):
        self.connection = _Connection()
        self.cursor = _Cursor()
        self.sql = {
            'cnxn': self.connection,
            'cursor': self.cursor,
            'sfbrange': 'stateframe-range',
        }
        self.error = None


def _run_fbin_block(disabled):
    _inc_time, _index, fbin_if = _fbin_if()
    wrapper = ast.FunctionDef(
        name='_run_fbin_block',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self'),
                                                  ast.arg(arg='data'),
                                                  ast.arg(arg='msg')],
                           vararg=None, kwonlyargs=[], kw_defaults=[],
                           kwarg=None, defaults=[]),
        body=[copy.deepcopy(fbin_if)],
        decorator_list=[],
    )
    module = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        'DISABLE_FBIN': disabled,
        'stateframedef': _StateframeDef,
    }
    exec(compile(module, _SOURCE_PATH, 'exec'), namespace)
    scheduler = _Scheduler()
    namespace['_run_fbin_block'](scheduler, b'raw', 'No Error')
    return scheduler


class ScheduleFbinGateTests(unittest.TestCase):
    def test_environment_flag_defaults_on_and_accepts_explicit_disable(self):
        self.assertFalse(_flag_value(None))
        self.assertTrue(_flag_value('1'))

    def test_default_path_inserts_and_disabled_path_does_not(self):
        enabled = _run_fbin_block(False)
        self.assertEqual(enabled.cursor.calls,
                         [('insert into fBin (Bin) values (?)',
                           b'transformed:raw')])
        self.assertEqual(enabled.connection.commits, 1)

        disabled = _run_fbin_block(True)
        self.assertEqual(disabled.cursor.calls, [])
        self.assertEqual(disabled.connection.commits, 0)

    def test_guard_contains_only_fbin_work_and_leaves_logging_outside(self):
        inc_time, index, fbin_if = _fbin_if()
        self.assertTrue(any('fBin' in value for value in _string_constants(fbin_if)))
        self.assertFalse(any('hBin' in value or 'sf_file' in value or 'sh_file' in value
                             for value in _string_constants(fbin_if)))
        self.assertTrue(any(isinstance(item, ast.Name) and item.id == 'DISABLE_FBIN'
                            for item in ast.walk(fbin_if.test)))
        self.assertIsInstance(inc_time.body[index + 1], ast.Assign)
        self.assertIn('sf_file', _string_constants(inc_time.body[index + 1]))


if __name__ == '__main__':
    unittest.main()
