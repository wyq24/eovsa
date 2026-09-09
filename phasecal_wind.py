"""Pure wind-telemetry decisions for the legacy phasecal scheduler."""

import math


# This signed WindScram value was observed in both calm and windy ACC
# samples.  Treat it as a legacy status value, while rejecting other errors.
WINDSCRAM_LEGACY_STATUS = -1950679035
MILLISECONDS_PER_DAY = 86400000.0
CONTROLLER_CLOCK_MAX_AGE_SECONDS = 1.0


def _finite(value):
    try:
        return not math.isnan(float(value)) and not math.isinf(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _integral_finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not _finite(number) or int(number) != number:
        return None
    return int(number)


def evaluate_controller_clocks(frame_mjd, crio_clock_ms,
                               system_clock_mjday, system_clock_ms,
                               max_age_seconds=CONTROLLER_CLOCK_MAX_AGE_SECONDS):
    """Check Ant16 controller clock freshness against the ACC frame MJD.

    ``cRIOClockms`` is day-relative, so compare it with the nearest of the
    current, previous, and following MJD days.  The antenna system clock has
    its own day field.  This keeps midnight rollover from looking like a
    controller outage while retaining the existing one-second freshness
    criterion used by ``sf_display``.
    """
    result = {
        'available': False,
        'reason': 'Ant16 controller clock invalid',
        'crio_age_seconds': None,
        'system_age_seconds': None,
    }
    try:
        frame = float(frame_mjd)
    except (TypeError, ValueError, OverflowError):
        return result
    if not _finite(frame):
        return result
    crio_ms = _integral_finite(crio_clock_ms)
    system_day = _integral_finite(system_clock_mjday)
    system_ms = _integral_finite(system_clock_ms)
    if (crio_ms is None or system_day is None or system_ms is None or
            crio_ms < 0 or crio_ms >= MILLISECONDS_PER_DAY or
            system_ms < 0 or system_ms >= MILLISECONDS_PER_DAY or
            system_day < 0):
        return result

    current_day = int(math.floor(frame))
    crio_candidates = [current_day - 1, current_day, current_day + 1]
    crio_mjd = min(
        (day + crio_ms / MILLISECONDS_PER_DAY for day in crio_candidates),
        key=lambda value: abs(value - frame))
    system_mjd = system_day + system_ms / MILLISECONDS_PER_DAY
    crio_age = abs(crio_mjd - frame) * 86400.0
    system_age = abs(system_mjd - frame) * 86400.0
    result['crio_age_seconds'] = crio_age
    result['system_age_seconds'] = system_age
    if crio_age > max_age_seconds or system_age > max_age_seconds:
        result['reason'] = 'Ant16 controller clock stale'
        return result
    result['available'] = True
    result['reason'] = 'Ant16 controller clocks fresh'
    return result


def evaluate_wind(wind_mph, sample_age_seconds, sample_time,
                  wind_limit_mph, scram_state, scram_comm_err,
                  stateframe_ok=True, acc_age_seconds=0.0,
                  acc_stale_limit_seconds=30.0,
                  stale_limit_seconds=300.0):
    """Evaluate whether an optional phase calibration should be skipped.

    :param wind_mph: Two-minute rolling average wind speed in mph.
    :param sample_age_seconds: Age of the weather sample at decision time.
    :param sample_time: Original weather sample timestamp for diagnostics.
    :param wind_limit_mph: Configured Ant 16 wind-scram threshold.
    :param scram_state: Ant 16 WindScram state value.
    :param scram_comm_err: Ant 16 WindScram communication error value.
    :param stateframe_ok: Whether the current ACC stateframe was obtained.
    :param acc_age_seconds: Age of the ACC stateframe root timestamp.
    :param acc_stale_limit_seconds: Maximum accepted ACC stateframe age.
    :param stale_limit_seconds: Maximum accepted weather sample age.
    :returns: A diagnostic dictionary containing ``skip`` and ``reason``.
    :rtype: dict
    """
    result = {
        'skip': False,
        'reason': 'weather unavailable',
        'wind_mph': wind_mph,
        'wind_limit_mph': wind_limit_mph,
        'sample_time': sample_time,
        'sample_age_seconds': sample_age_seconds,
        'acc_age_seconds': acc_age_seconds,
        'wind_confirmed': False,
        'weather_available': False,
        'scram_reliable': False,
        'scram_state': scram_state,
        'scram_comm_err': scram_comm_err,
    }

    if not stateframe_ok:
        result['reason'] = 'ACC stateframe unavailable'
        result['skip'] = True
        return result
    try:
        comm_err_float = float(scram_comm_err)
        scram_float = float(scram_state)
        if (not _finite(comm_err_float) or not _finite(scram_float) or
                int(comm_err_float) != comm_err_float or
                int(scram_float) != scram_float):
            raise ValueError
        comm_err = int(comm_err_float)
        scram = int(scram_float)
    except (TypeError, ValueError, OverflowError):
        result['reason'] = 'ACC wind telemetry invalid'
        result['skip'] = True
        return result
    if comm_err not in (0, WINDSCRAM_LEGACY_STATUS):
        result['reason'] = 'ACC wind telemetry communication error'
        result['skip'] = True
        return result

    try:
        acc_age = float(acc_age_seconds)
    except (TypeError, ValueError, OverflowError):
        acc_age = -1.0
    if (not _finite(acc_age) or acc_age < 0.0 or
            acc_age > float(acc_stale_limit_seconds)):
        result['reason'] = 'ACC stateframe stale'
        result['skip'] = True
        return result
    result['acc_age_seconds'] = acc_age
    result['scram_reliable'] = True
    result['scram_state'] = scram
    result['scram_comm_err'] = comm_err

    weather_valid = False
    wind = age = None
    limit = None
    try:
        wind = float(wind_mph)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        age = float(sample_age_seconds)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        limit = float(wind_limit_mph)
    except (TypeError, ValueError, OverflowError):
        pass
    if (_finite(limit) and limit >= 0.0 and _finite(wind) and
            _finite(age) and wind >= 0.0 and age >= 0.0 and
            age <= float(stale_limit_seconds)):
        weather_valid = True
        result['wind_mph'] = wind
        result['sample_age_seconds'] = age
        result['wind_limit_mph'] = limit
    result['weather_available'] = weather_valid

    # A reliable active WindScram is sufficient to skip the pair, even when
    # the weather station is unavailable.  Evaluate weather independently
    # first so the decision log reports valid weather alongside the scram.
    if scram != 0:
        result['skip'] = True
        result['wind_confirmed'] = True
        result['reason'] = 'wind scram active'
        return result

    if not weather_valid:
        if not _finite(limit) or limit < 0.0:
            result['reason'] = 'wind limit invalid'
            result['skip'] = True
        else:
            result['reason'] = 'weather unavailable'
            result['skip'] = False
        return result
    if wind >= limit:
        result['skip'] = True
        result['wind_confirmed'] = True
        result['reason'] = 'wind at or above limit'
        return result

    result['skip'] = False
    result['reason'] = 'wind within limit'
    return result
