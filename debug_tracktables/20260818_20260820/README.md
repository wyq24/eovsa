# EOVSA debug tracktables: 2026-08-18 through 2026-08-20

This bundle contains every `.radec` table produced by an expanded
`$MK_TABLES` command in the standard solar schedules for the three requested
dates.  The scheduler normally overwrites `sun_tab.radec` and
`pcal_tab.radec`; these copies have unique names so that every invocation is
preserved for debugging.

## Contents

- `tables/<YYYYMMDD>/`: 20 uniquely named tracktables per date, 60 total.
- `schedules/<YYYYMMDD>.scd`: the 31-line generated schedule used for each date.
- `manifest.csv`: schedule line, next line, expanded `$MK_TABLES` command,
  source, runtime filename, saved filename, MJD range, line count, and SHA-256.
- `ctl_snapshot/`: the `.ctl` files copied read-only from the live scheduler
  asset directory on helios.
- `expanded_ctl/`: the fully expanded macro text for each of the 60
  `$MK_TABLES` invocations, retained for line-by-line debugging.
- `source_snapshot/`: the deployed `whenup.py` and `eovsa_tracktable.py` source
  used for generation.
- `generate_tracktables.py`: the local, side-effect-free reproduction script.

The source snapshots match `/common/python/current` on helios:

- `whenup.py`: `767b58cc3fa7cb02563c000f8f02fb42d00f2830bc57e8da88a124239d78a754`
- `eovsa_tracktable.py`: `bcab5c664a1dcfd04dfd85662a5052f05d08cab46ac43a532aa37f0fb6faa48e`

## Regenerate locally

From the `repos/eovsa` directory:

```text
/Users/fisher/.pyenv/versions/3.10.13/bin/python \
  debug_tracktables/20260818_20260820/generate_tracktables.py
```

The script converts the frozen Python 2 source in memory, supplies the known
working EOVSA Python 3.10 dependencies, expands macro arguments exactly as
`schedule.py` does, and calls the deployed `make_tracktable()` algorithm with
the current and next schedule timestamps.

It does not run `schedule.py`, execute `.ctl` macros, connect to the ACC, FTP
files, or send control commands.  It only rewrites this local debug bundle.

The bundle covers `$MK_TABLES` products only.  SOLPNT trajectory-pattern files
created by `$MK_TRAJ` are separate artifacts and are not included.

An independent worker generation produced the same 60 tracktables.  Every
file was compared byte-for-byte with the files under `tables/`; all matched.

## Runtime note

A fresh `/common/anaconda2/bin/python` process started from
`/home/sched/Dropbox/PythonCode/Current` on helios currently fails while
loading `aipy` because of a NumPy binary mismatch.  The live scheduler was
confirmed to already be running with that interpreter and working directory;
it was not restarted or otherwise touched.  Local generation was therefore
used to avoid interacting with the observation system.
