#!/usr/bin/env python3
"""Generate debug copies of every $MK_TABLES product for three schedules."""

import csv
import hashlib
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy
from lib2to3.refactor import RefactoringTool, get_fixers_from_package


DATES = ("2026-08-18", "2026-08-19", "2026-08-20")
BUNDLE_DIR = Path(__file__).resolve().parent
REPOS_DIR = BUNDLE_DIR.parents[2]
EOVSA_DIR = REPOS_DIR / "eovsa"
EOVSAPY_DIR = REPOS_DIR / "eovsapy"
CTL_DIR = BUNDLE_DIR / "ctl_snapshot"
SOURCE_DIR = BUNDLE_DIR / "source_snapshot"
SCHEDULE_DIR = BUNDLE_DIR / "schedules"
TABLE_DIR = BUNDLE_DIR / "tables"


def _git_source(filename):
    """Return the committed legacy source that matches the helios deployment."""
    return subprocess.check_output(
        ["git", "show", "HEAD:" + filename],
        cwd=str(EOVSA_DIR),
    )


def _snapshot_source(filename):
    """Save and return a reproducible copy of one legacy source file."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / filename
    if not path.exists():
        path.write_bytes(_git_source(filename))
    return path.read_bytes().decode("utf-8")


def _convert_legacy(source, filename):
    """Convert Python 2 syntax without rewriting the injected imports."""
    if not source.endswith("\n"):
        source += "\n"
    fixers = [
        fixer
        for fixer in get_fixers_from_package("lib2to3.fixes")
        if not fixer.endswith(("fix_import", "fix_imports", "fix_imports2"))
    ]
    return str(RefactoringTool(fixers).refactor_string(source, filename))


def _load_legacy_modules():
    """Load the deployed schedule and tracktable algorithms under Python 3.10."""
    sys.path.insert(0, str(EOVSAPY_DIR))
    from eovsapy import eovsa_array, eovsa_cat, eovsa_visibility, util

    sys.modules["util"] = util
    sys.modules["eovsa_array"] = eovsa_array
    sys.modules["eovsa_cat"] = eovsa_cat
    sys.modules["eovsa_visibility"] = eovsa_visibility

    whenup = types.ModuleType("legacy_whenup")
    source = _convert_legacy(_snapshot_source("whenup.py"), "whenup.py")
    exec(compile(source, "legacy_whenup.py", "exec"), whenup.__dict__)

    class NumpyCompat:
        """Restore the ragged-array behavior used by the deployed NumPy."""

        def __getattr__(self, name):
            return getattr(numpy, name)

        def array(self, value, *args, **kwargs):
            try:
                return numpy.array(value, *args, **kwargs)
            except ValueError:
                kwargs["dtype"] = object
                return numpy.array(value, *args, **kwargs)

    whenup.np = NumpyCompat()

    tracktable = types.ModuleType("legacy_eovsa_tracktable")
    source = _convert_legacy(
        _snapshot_source("eovsa_tracktable.py"), "eovsa_tracktable.py"
    )
    exec(compile(source, "legacy_eovsa_tracktable.py", "exec"), tracktable.__dict__)
    return util, eovsa_cat, whenup, tracktable


def _expand_macro(line, ctl_line):
    """Apply schedule.py's single #N macro-argument substitution."""
    commands = line[20:].split()
    if "#" in ctl_line[1:]:
        hash_index = ctl_line[1:].find("#") + 1
        argument_index = int(ctl_line[hash_index + 1 : hash_index + 2])
        ctl_line = (
            ctl_line[:hash_index]
            + commands[argument_index]
            + ctl_line[hash_index + 2 :]
        )
    return ctl_line.rstrip("\n")


def _table_events(lines):
    """Yield all expanded $MK_TABLES calls with scheduler time ranges."""
    for schedule_index, line in enumerate(lines[:-1]):
        commands = line[20:].split()
        if not commands or commands[0].upper().startswith("FEMPOWER"):
            continue
        ctl_path = CTL_DIR / (commands[0] + ".ctl")
        if not ctl_path.exists():
            continue
        for raw_ctl_line in ctl_path.read_text().splitlines(True):
            ctl_line = _expand_macro(line, raw_ctl_line)
            tokens = ctl_line.split()
            if tokens and tokens[0].upper() == "$MK_TABLES":
                if len(tokens) != 3:
                    raise RuntimeError("Unexpected $MK_TABLES line: " + ctl_line)
                yield schedule_index, line, lines[schedule_index + 1], ctl_line


def _slug(value):
    """Return a filename-safe representation of one manifest value."""
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", value)


def _sha256(text):
    """Return the SHA-256 digest of generated table text."""
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def main():
    """Generate schedules, uniquely named tables, and a CSV manifest."""
    util, eovsa_cat, whenup, tracktable = _load_legacy_modules()
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    antenna_array = eovsa_cat.eovsa_array_with_cat()
    manifest_rows = []

    for date in DATES:
        date_key = date.replace("-", "")
        lines = whenup.make_sched(
            t=util.Time(date + " 12:00:00"), ant13_fem_power=True
        )
        (SCHEDULE_DIR / (date_key + ".scd")).write_text("\n".join(lines) + "\n")
        date_table_dir = TABLE_DIR / date_key
        date_table_dir.mkdir(parents=True, exist_ok=True)

        for sequence, event in enumerate(_table_events(lines), 1):
            schedule_index, line, next_line, ctl_line = event
            _, runtime_name, source_name = ctl_line.split()
            mjd1 = util.Time(line[:19]).mjd
            mjd2 = util.Time(next_line[:19]).mjd
            table = tracktable.make_tracktable(
                source_name, antenna_array, mjd1, mjd2
            )
            if not isinstance(table, str):
                raise RuntimeError("Tracktable generation failed: {!r}".format(table))

            timestamp = line[:19].replace("-", "").replace(":", "").replace(" ", "T")
            macro = line[20:].split()[0].upper()
            filename = "{:02d}_{}_{}_{}_{}.radec".format(
                sequence,
                timestamp,
                _slug(macro),
                _slug(source_name),
                _slug(runtime_name),
            )
            saved_path = date_table_dir / filename
            saved_path.write_text(table)
            manifest_rows.append(
                {
                    "schedule_date": date_key,
                    "sequence": sequence,
                    "schedule_line_number": schedule_index + 1,
                    "schedule_line": line,
                    "next_schedule_line": next_line,
                    "macro": macro,
                    "expanded_mk_tables": ctl_line,
                    "source": source_name,
                    "runtime_filename": runtime_name + ".radec",
                    "saved_filename": str(saved_path.relative_to(BUNDLE_DIR)),
                    "mjd1": "{:.10f}".format(mjd1),
                    "mjd2": "{:.10f}".format(mjd2),
                    "table_lines": len(table.splitlines()),
                    "sha256": _sha256(table),
                }
            )

    fieldnames = list(manifest_rows[0])
    with (BUNDLE_DIR / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print("Generated {} tracktables".format(len(manifest_rows)))


if __name__ == "__main__":
    main()
