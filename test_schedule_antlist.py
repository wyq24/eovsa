"""Focused tests for validated scheduler subarray arguments."""

import ast
import os
import re
import tempfile
import unittest
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
from numpy import array


_SOURCE_PATH = os.path.join(os.path.dirname(__file__), 'schedule.py')
_UTIL_SOURCE_PATH = os.path.join(os.path.dirname(__file__), 'util.py')
_REFACTOR = RefactoringTool(get_fixers_from_package('lib2to3.fixes'))
_FUNCTION_NAMES = (
    'get_antlist', '_parse_subarray_ant_tokens', '_subarray_exclude',
    'resolve_subarray_args')


def _extract_functions(path, names, namespace):
    with open(path) as source_file:
        source = source_file.read()
    if not source.endswith('\n'):
        source += '\n'
    source = str(_REFACTOR.refactor_string(source, path))
    tree = ast.parse(source)
    functions = [item for item in tree.body
                 if isinstance(item, ast.FunctionDef) and item.name in names]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, path, 'exec'), namespace)


def _extract_constant(path, name, namespace):
    with open(path) as source_file:
        source = source_file.read()
    if not source.endswith('\n'):
        source += '\n'
    source = str(_REFACTOR.refactor_string(source, path))
    tree = ast.parse(source)
    assignment = next(item for item in tree.body
                      if isinstance(item, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == name
                          for target in item.targets))
    module = ast.Module(body=[assignment], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, path, 'exec'), namespace)


def _get_ant_str2list():
    namespace = {'array': array}
    _extract_functions(_UTIL_SOURCE_PATH, ('ant_str2list',), namespace)
    return namespace['ant_str2list']


def _get_resolver():
    class _Util(object):
        pass
    util = _Util()
    util.ant_str2list = _get_ant_str2list()
    namespace = {'util': util, 're': re}
    _extract_constant(_SOURCE_PATH, '_SUBARRAY_ANT_TOKEN', namespace)
    _extract_functions(_SOURCE_PATH, _FUNCTION_NAMES, namespace)
    return namespace['resolve_subarray_args']


def _write_antlist(contents):
    handle, path = tempfile.mkstemp(prefix='antlist-', suffix='.antlist')
    os.close(handle)
    with open(path, 'w') as output:
        output.write(contents)
    return path


class ScheduleAntlistTests(unittest.TestCase):
    def setUp(self):
        self.resolve = _get_resolver()
        self.paths = []

    def tearDown(self):
        for path in self.paths:
            os.unlink(path)

    def _named(self, contents, *args):
        path = _write_antlist(contents)
        self.paths.append(path)
        return self.resolve([path] + list(args))

    def test_named_exclusion_uses_current_antlist_and_normalizes_commas(self):
        contents = 'sun ant1, ant3-6 ant8-11 ant13 ant15\n'
        self.assertEqual(
            self._named(contents, 'sun', 'EXCLUDE', 'ant3-4', 'ant15'),
            'ant1 ant5 ant6 ant8 ant9 ant10 ant11 ant13')

    def test_named_without_exclusion_is_unchanged(self):
        self.assertEqual(self._named('sun ant1 ant3-4\n', 'sun'),
                         'ant1 ant3-4')

    def test_direct_exclusion_and_already_absent_are_safe(self):
        self.assertEqual(
            self.resolve(['ant1', 'ant3-5', 'ant8', 'exclude', 'ant3', 'ant15']),
            'ant1 ant4 ant5 ant8')

    def test_invalid_exclusions_raise_before_send(self):
        contents = 'sun ant1 ant3-6 ant8\n'
        invalid = (
            ('sun', 'exclude'),
            ('sun', 'exclude', 'ant4-ant3'),
            ('sun', 'exclude', 'ant17'),
            ('sun', 'exclude', 'foo'),
            ('sun', 'unexpected', 'ant3'),
            ('sun', 'exclude', 'ant1-8'),
        )
        for args in invalid:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    self._named(contents, *args)

    def test_unknown_extra_and_multiple_exclusions_raise(self):
        contents = 'sun ant1 ant3-6 ant8\n'
        with self.assertRaises(ValueError):
            self._named(contents, 'sun', 'exclude', 'ant3', 'extra')
        with self.assertRaises(ValueError):
            self._named(contents, 'sun', 'exclude', 'ant3', 'exclude', 'ant4')

    def test_exclusion_cannot_empty_the_selection(self):
        with self.assertRaises(ValueError):
            self._named('sun ant1-3\n', 'sun', 'exclude', 'ant1-3')

    def test_missing_named_list_raises(self):
        with self.assertRaises(ValueError):
            self._named('sun ant1\n', 'missing', 'exclude', 'ant3')

    def test_unreadable_antlist_raises(self):
        with self.assertRaises(ValueError):
            self.resolve(['/tmp/does-not-exist-eovsa-antlist.antlist', 'sun',
                          'exclude', 'ant3'])


if __name__ == '__main__':
    unittest.main()
