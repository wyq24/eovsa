"""Behavior tests for optional CelesTrak catalog loading."""

import importlib
import os
import tempfile
import sys
import types
import unittest

import numpy


class _Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_eovsa_cat():
    """Import eovsa_cat with small dependency stand-ins for catalog tests."""
    if not hasattr(numpy, 'mat'):
        numpy.mat = numpy.asmatrix
    aipy = types.ModuleType('aipy')

    class _RadioBody(object):
        def __init__(self, *args, **kwargs):
            pass

    class _RadioFixedBody(object):
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', args[-1] if args else '')

    class _SrcCatalog(list):
        def compute(self, aa):
            pass

    aipy.phs = _Namespace(RadioBody=_RadioBody)
    aipy.amp = _Namespace(
        RadioFixedBody=_RadioFixedBody,
        RadioSpecial=_RadioFixedBody,
        SrcCatalog=_SrcCatalog)
    sys.modules['aipy'] = aipy
    sys.modules['ephem'] = types.ModuleType('ephem')
    readvla = types.ModuleType('readvla')
    readvla.readvlacaldb = lambda: []
    sys.modules['readvla'] = readvla
    eovsa_array = types.ModuleType('eovsa_array')
    eovsa_array.eovsa_array = lambda: object()
    sys.modules['eovsa_array'] = eovsa_array
    sys.modules['urllib2'] = types.ModuleType('urllib2')
    sys.modules.pop('eovsa_cat', None)
    return importlib.import_module('eovsa_cat')


class SatelliteCatalogOptionTest(unittest.TestCase):

    def setUp(self):
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in ('aipy', 'ephem', 'readvla', 'eovsa_array',
                         'urllib2', 'eovsa_cat')}
        self._had_numpy_mat = hasattr(numpy, 'mat')

    def tearDown(self):
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if not self._had_numpy_mat and hasattr(numpy, 'mat'):
            del numpy.mat

    def test_satellites_are_explicitly_optional(self):
        eovsa_cat = _load_eovsa_cat()
        calls = []
        eovsa_cat.load_VLAcals = lambda: calls.append('vla') or []
        eovsa_cat.load_sidereal_cats = lambda: calls.append('sidereal') or []
        eovsa_cat.load_geosats = lambda: calls.append('geo') or []
        eovsa_cat.load_o3bsats = lambda: calls.append('o3b') or []
        eovsa_cat.load_gpssats = lambda: calls.append('gps') or []

        eovsa_cat.load_cat(include_satellites=False)
        self.assertEqual(calls, ['vla', 'sidereal'])

        calls[:] = []
        eovsa_cat.load_cat()
        self.assertEqual(calls, ['vla', 'geo', 'sidereal', 'o3b', 'gps'])

    def test_scd_detection_uses_third_field_only(self):
        eovsa_cat = _load_eovsa_cat()
        self.assertTrue(eovsa_cat.schedule_uses_geosats([
            '2026-08-30 01:24:00 GEOSAT ECHOSTAR_11 kband.fsq']))
        self.assertTrue(eovsa_cat.schedule_uses_geosats([
            '2026-08-30 01:24:00 DELAYCAL CIEL-2 band23.fsq']))
        self.assertFalse(eovsa_cat.schedule_uses_geosats([
            '2026-08-30 01:24:00 PHASECAL GEOSAT pcal.fsq']))
        self.assertFalse(eovsa_cat.schedule_uses_geosats([
            '# 2026-08-30 01:24:00 GEOSAT fake']))

    def test_array_catalog_option_is_forwarded(self):
        eovsa_cat = _load_eovsa_cat()
        calls = []

        class _Catalog(object):
            def compute(self, aa):
                calls.append(('compute', aa))

        eovsa_cat.load_cat = lambda **kwargs: calls.append(kwargs) or _Catalog()
        class _Array(object):
            pass

        array = _Array()
        eovsa_cat.eovsa_array = lambda: array
        self.assertIs(eovsa_cat.eovsa_array_with_cat(include_satellites=False), array)
        self.assertEqual(calls[0], {'include_satellites': False})
        self.assertEqual(calls[1], ('compute', array))
        calls[:] = []
        self.assertIs(eovsa_cat.eovsa_array_with_cat(), array)
        self.assertEqual(calls[0], {'include_satellites': True})

    def test_execution_cache_reader_never_refreshes(self):
        eovsa_cat = _load_eovsa_cat()
        eovsa_cat.urllib2.urlopen = lambda *args, **kwargs: self.fail(
            'execution-side cache read attempted a network request')
        fd, cache_path = tempfile.mkstemp()
        try:
            os.write(fd, b'ECHOSTAR_11\nline 2\nline 3\n')
            os.close(fd)
            self.assertEqual(eovsa_cat.read_cached_text(cache_path), [
                'ECHOSTAR_11\n', 'line 2\n', 'line 3\n'])
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            os.unlink(cache_path)


if __name__ == '__main__':
    unittest.main()
