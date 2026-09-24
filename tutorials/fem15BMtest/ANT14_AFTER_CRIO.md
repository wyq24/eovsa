# Ant14 test after the cRIO coefficients are loaded

## 1. Get the test code

```bash
cd /home/sched/Dropbox/PythonCode/Current
git fetch origin
git checkout agent/ant14-fem-display-support
git pull --ff-only
```

## 2. Install the schedule files

```bash
cp --no-clobber tutorials/fem15BMtest/FEM14FIELDTEST.ctl .
cp --no-clobber tutorials/fem15BMtest/FEM14SAFE.ctl .
cp tutorials/fem15BMtest/FEM14FIELDTEST.scd.template FEM14FIELDTEST.scd
nano FEM14FIELDTEST.scd
```

Replace the time in `FEM14FIELDTEST.scd` with a future local time.

## 3. Run the test

Start the logger:

```bash
python tutorials/fem15BMtest/fem15_stateframe_logger.py \
  --ant 14 --duration 420 \
  --output /tmp/fem14_fieldtest.csv
```

In a seperate terminal:

```bash
python sf_display.py
```

Load `FEM14FIELDTEST.scd` in the scheduler and press `GO`. If something look
wrong, abort and run:

```text
FEM14SAFE
```

## 4. Fit the Ant14 model

Run this after the test. The default voltage window removes the low voltage
floor and stays below saturation. Change `VMIN` or `VMAX` if the test plot show
a different valid range.

```bash
python - <<'PY'
import csv
import numpy as np

CSV_FILE = "/tmp/fem14_fieldtest.csv"
VMIN = 0.05
VMAX = 1.05

with open(CSV_FILE, "r") as input_file:
    rows = list(csv.DictReader(input_file))

fits = {}
for pol in ("h", "v"):
    for nd_state in (0, 1):
        attenuation = []
        power = []
        for row in rows:
            raw_nd = str(row["nd"]).strip().lower()
            row_nd = 1 if raw_nd in ("1", "true", "on") else 0
            voltage = float(row[pol + "_voltage"])
            measured_power = float(row[pol + "_power"])
            if row_nd != nd_state:
                continue
            if not np.isfinite(voltage) or not np.isfinite(measured_power):
                continue
            if not VMIN < voltage <= VMAX:
                continue
            total_attn = float(row[pol + "_attn1"]) + float(row[pol + "_attn2"])
            attenuation.append(total_attn)
            power.append(measured_power)

        if len(attenuation) < 3:
            raise RuntimeError("Not enough valid %s ND=%d rows" % (pol, nd_state))
        slope, intercept = np.polyfit(attenuation, power, 1)
        fits[(pol.upper(), nd_state)] = (float(slope), float(intercept))

print("COEFF_SLOPE[14] = {")
for pol in ("H", "V"):
    print('    "%s": {"OFF": %.12g, "ON": %.12g},' % (
        pol, fits[(pol, 0)][0], fits[(pol, 1)][0]))
print("}")
print("COEFF_INTERCEPT[14] = {")
print('    "lab": {')
for pol in ("H", "V"):
    print('        "%s": {"OFF": %.12g, "ON": %.12g},' % (
        pol, fits[(pol, 0)][1], fits[(pol, 1)][1]))
print("    },")
print("}")
print('VOLTAGE_THRESHOLD[14] = {"H": 1.105, "V": 1.105}')
PY
```

Copy the printed block below the constant definitions in
`attn_power_model.py`:

```bash
nano attn_power_model.py
```

## 5. Test and check SF Display

```bash
python -m unittest -v test_attn_power_model
python sf_display.py
```

Above the threshold, the Ant14 modeled power should have `*`. Below the
threshold it should show the measured cRIO power without `*`.

## 6. Commit and push

```bash
git add attn_power_model.py
git commit -m "Add Ant14 FEM power calibration"
git push
```
