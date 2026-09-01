"""Pure helpers for exporting the scheduler's public schedule status."""

import hashlib
import json
import os
import tempfile


try:
    text_type = unicode
except NameError:
    text_type = str


try:
    binary_type = bytes
except NameError:
    binary_type = str


def _text(value):
    if isinstance(value, text_type):
        return value
    return text_type(value)


def _line(value):
    return _text(value).rstrip('\r\n')


def _schedule_text(lines):
    return ''.join(_line(value) + '\n' for value in lines)


def schedule_sha256(lines):
    payload = _schedule_text(lines)
    if not isinstance(payload, binary_type):
        payload = payload.encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _remaining_line_counts(lines):
    counts = {}
    for value in lines:
        value = _line(value)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _valid_executed_lines(lines, executed_lines):
    available = _remaining_line_counts(lines)
    valid = []
    for value in executed_lines or []:
        value = _line(value)
        if available.get(value, 0) > 0:
            valid.append(value)
            available[value] -= 1
    return valid


def build_status_lines(lines, current_index=None, executed_lines=None,
                       planned_indices=None):
    """Return feed lines with current and Skip Phacal prefixes.

    ``executed_lines`` is intentionally separate from the scheduler's generic
    ``Skipped`` GUI status.  Duplicate schedule lines are matched by count so
    an exact executed-line record cannot gray every duplicate occurrence.
    """
    remaining_executed = _remaining_line_counts(executed_lines or [])
    planned = set(planned_indices or ())
    output = []
    for index, value in enumerate(lines):
        value = _line(value)
        if current_index is not None and index == current_index:
            prefix = '* '
            if remaining_executed.get(value, 0) > 0:
                remaining_executed[value] -= 1
        elif remaining_executed.get(value, 0) > 0:
            prefix = 'S '
            remaining_executed[value] -= 1
        elif index in planned:
            prefix = 'S '
        else:
            prefix = '  '
        output.append(prefix + value)
    return output


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path) or '.'
    prefix = '.%s.' % os.path.basename(path)
    fd, temporary_path = tempfile.mkstemp(prefix=prefix, dir=directory)
    try:
        with os.fdopen(fd, 'wb') as handle:
            encoded = json.dumps(payload, sort_keys=True,
                                 separators=(',', ':'))
            if not isinstance(encoded, binary_type):
                encoded = encoded.encode('utf-8')
            handle.write(encoded + b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def load_executed_lines(path, lines):
    """Load executed Skip Phacal rows only when the schedule hash matches."""
    try:
        with open(path, 'rb') as handle:
            payload = json.load(handle)
    except (IOError, OSError, ValueError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []
    if payload.get('version') != 1:
        return []
    if payload.get('schedule_sha256') != schedule_sha256(lines):
        return []
    executed_lines = payload.get('executed_skip_phacal_lines')
    if not isinstance(executed_lines, list):
        return []
    return _valid_executed_lines(lines, executed_lines)


def write_executed_lines(path, lines, executed_lines):
    """Atomically persist executed Skip Phacal rows for this schedule."""
    payload = {
        'version': 1,
        'schedule_sha256': schedule_sha256(lines),
        'executed_skip_phacal_lines': _valid_executed_lines(
            lines, executed_lines),
    }
    _atomic_write_json(path, payload)


def write_schedule_status(status_path, state_path, lines, current_index=None,
                          executed_lines=None, planned_indices=None):
    """Persist sidecar state and rewrite the public text feed."""
    lines = [_line(value) for value in lines]
    executed_lines = _valid_executed_lines(lines, executed_lines or [])
    write_executed_lines(state_path, lines, executed_lines)
    output = build_status_lines(
        lines,
        current_index=current_index,
        executed_lines=executed_lines,
        planned_indices=planned_indices,
    )
    with open(status_path, 'w') as handle:
        for value in output:
            handle.write(value + '\n')
