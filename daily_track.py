if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import os
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except:
        user_paths = []
    print(user_paths)

import argparse
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import numpy.ma as ma
# from astropy.time import Time
import pipeline_cal as pc
from util import Time
from glob import glob
import shutil
import dbutil as db
import matplotlib.patheffects as pe




def annotate_observing_intervals(
        ax,
        SOS,                  # 1D array-like of start times in Matplotlib date numbers
        EOS,                  # 1D array-like of end times in Matplotlib date numbers, same length as SOS
        projects,             # 1D array-like of project strings, same length as SOS
        alpha_span=0.18,
        lw_edge=0.6,
        min_label_minutes=8,   # don't label intervals shorter than this (minutes)
        lanes=3,               # stagger text across N vertical lanes per category
        fontsize=8):
    """
    Cleaner, de-cluttered annotation:
      - merge consecutive scans of same category
      - draw semi-transparent spans
      - place labels at midpoints, staggered in lanes
      - skip very short intervals
    """
    artists = {'spans': [], 'labels': [], 'start_vlines': [], 'end_vlines': [], 'final_end': []}
    if len(SOS) == 0:
        return artists

    # --- normalize project names ----
    norm = np.char.upper(np.char.replace(projects.copy().astype(str), ' ', ''))
    is_sun   = (norm == 'NORMALOBSERVING')
    is_idle  = (norm == 'NONE')
    prefixes = np.char.array([s[:4] for s in norm])
    is_cal   = (prefixes == 'GAIN') | (prefixes == 'PHAS') | (prefixes == 'SOLP') | (norm == 'SOLPNTCAL')
    is_other = ~(is_sun | is_idle | is_cal)

    # --- category palette (soft, readable) ---
    # spans (fills)
    span_colors = {
        'SUN'  : '#9ecae1',  # light blue
        'CAL'  : '#c7e9c0',  # light green
        'OTHER': '#fdd0a2',  # light orange
        'IDLE' : '#e0e0e0',  # light gray
    }
    # label color
    label_color = '#262626'

    # --- y positions (3 lanes per category, near top) ---
    y0, y1 = ax.get_ylim()
    base_y = {
        'SUN'  : y1 * 0.925,
        'CAL'  : y1 * 0.945,
        'OTHER': y1 * 0.965,
        'IDLE' : y1 * 0.985,
    }
    lane_dy = (y1 - y0) * 0.006  # vertical separation between lanes

    # helper to map booleans to category key
    def _cat(i):
        if is_sun[i]:   return 'SUN'
        if is_idle[i]:  return 'IDLE'
        if is_cal[i]:   return 'CAL'
        return 'OTHER'

    SOS = np.asarray(SOS, dtype=float)
    EOS = np.asarray(EOS, dtype=float)
    n   = len(SOS)

    # --- merge contiguous blocks with same category ---
    blocks = []  # list of (start, end, category, label_text)
    if n:
        cur_s = SOS[0]
        cur_e = EOS[0]
        cur_c = _cat(0)
        cur_label = 'SUN' if cur_c == 'SUN' else (
            'IDLE' if cur_c == 'IDLE' else (
                'TPCAL' if norm[0] == 'SOLPNTCAL' else (
                    'GCAL' if norm[0].startswith('GAIN') else (
                        'PCAL' if norm[0].startswith('PHAS') else
                        projects[0]))))

        for i in range(1, n):
            c = _cat(i)
            # treat immediate adjacency as contiguous; you can allow tiny gaps if desired
            if (c == cur_c) and np.isclose(SOS[i], cur_e, rtol=0, atol=2.0/1440.0):  # <= 2 minutes gap OK
                # extend the current block
                cur_e = EOS[i]
            else:
                blocks.append((cur_s, cur_e, cur_c, cur_label))
                cur_s = SOS[i]
                cur_e = EOS[i]
                cur_c = c
                cur_label = 'SUN' if c == 'SUN' else (
                    'IDLE' if c == 'IDLE' else (
                        'TPCAL' if norm[i] == 'SOLPNTCAL' else (
                            'GCAL' if norm[i].startswith('GAIN') else (
                                'PCAL' if norm[i].startswith('PHAS') else
                                projects[i]))))
        blocks.append((cur_s, cur_e, cur_c, cur_label))

    # --- draw spans + labels, stagger text in lanes, skip tiny blocks ---
    lane_counter = {'SUN': 0, 'CAL': 0, 'OTHER': 0, 'IDLE': 0}
    min_span = min_label_minutes / 1440.0  # days

    for (xs, xe, cat, label) in blocks:
        width = xe - xs
        # span
        span = ax.axvspan(xs, xe,
                          facecolor=span_colors[cat],
                          edgecolor='k',
                          linewidth=lw_edge,
                          alpha=alpha_span,
                          zorder=1)
        artists['spans'].append(span)

        # label (skip very short)
        if width >= min_span:
            lane = lane_counter[cat] % max(1, lanes)
            ytxt = base_y[cat] - lane * lane_dy
            xm   = xs + 0.5 * width
            txt = ax.text(
                xm, ytxt, str(label),
                ha='center', va='top', fontsize=fontsize,
                color=label_color, clip_on=True, zorder=3,
                path_effects=[pe.withStroke(linewidth=2, foreground='white', alpha=0.8)]
            )
            artists['labels'].append(txt)
            lane_counter[cat] += 1

    # optional: subtle start/end ticks (very light)
    vline_c = '#bdbdbd'
    artists['start_vlines'].append(ax.vlines(SOS, y0, y1, colors=vline_c, linewidth=0.3, zorder=0))
    artists['end_vlines'].append(ax.vlines(EOS, y0, y1, colors=vline_c, linewidth=0.3, linestyles='--', zorder=0))
    artists['final_end'].append(ax.vlines([EOS[-1]], y0, y1, colors=vline_c, linewidth=0.5, linestyles='--', zorder=0))

    ax.set_ylim(y0, y1)
    return artists

def get_projects(t):
    ''' Read all projects from SQL for the current date and return a summary
        as a dictionary with keys Timestamp, Project, and EOS (another timestamp)
    '''

    # timerange is 12 UT to 12 UT on next day, relative to the day in Time() object t
    trange = Time([int(t.mjd) + 11. / 24, int(t.mjd) + 37. / 24], format='mjd')
    tstart, tend = trange.lv.astype('str')
    cursor = db.get_cursor()
    mjd = t.mjd
    # Get the project IDs for scans during the period
    verstrh = db.find_table_version(cursor, trange[0].lv, True)
    if verstrh is None:
        print('No scan_header table found for given time.')
        return {}
    query = 'select Timestamp,Project from hV' + verstrh + '_vD1 where Timestamp between ' + tstart + ' and ' + tend + ' order by Timestamp'
    projdict, msg = db.do_query(cursor, query)
    if msg != 'Success':
        print(msg)
        return {}
    elif len(projdict) == 0:
        # No Project ID found, so return data and empty projdict dictionary
        print('SQL Query was valid, but no Project data were found.')
        return {}
    projdict['Timestamp'] = projdict['Timestamp'].astype('float')  # Convert timestamps from string to float
    for i in range(len(projdict['Project'])): projdict['Project'][i] = projdict['Project'][i].replace('\x00', '')
    # projdict.update({'EOS': projdict['Timestamp'][1:]})
    # projdict.update({'Timestamp': projdict['Timestamp'][:-1]})
    # projdict.update({'Project': projdict['Project'][:-1]})
    cursor.close()
    return projdict

def build_timerange_daily(date_str=None):
    """
    Build [start_dt, end_dt] in UTC datetimes.
    If date_str is None -> use 'yesterday' (UTC) as base date.
    Window: 12:00 UT base date -> 04:00 UT next day.
    """
    if date_str is None:
        dt_now = datetime.utcnow()
        base_dt = dt_now.date()
        if dt_now.hour < 7:
            base_dt -= timedelta(days=1)
    else:
        base_dt = datetime.strptime(date_str, '%Y-%m-%d').date()
    start_dt = datetime.combine(base_dt, datetime.min.time()) + timedelta(hours=12)
    if date_str is None:
        end_dt = dt_now
    else:
        end_dt = start_dt + timedelta(hours=16)
    return [start_dt, end_dt]

def adjust_eos_per_antenna0(t_plot, req_el_1d, SOS, EOS0,
                           step_deg=3.0,      # "big step" threshold (deg)
                           flat_eps=0.2,     # how flat is "flat" (deg)
                           flat_n=50):         # consecutive samples to call it flat
    """
    Returns EOS_adj (len=N scans) for ONE antenna.

    Logic:
      For each scan n, search within [SOS[n], EOS0[n]] for the last large step.
      From there, find the earliest time where a run of `flat_n` diffs stays within `flat_eps`.
      If found, set EOS_adj[n] to that timestamp; otherwise keep EOS0[n].
      Ensures EOS_adj[n] < SOS[n+1] (if exists).
    """
    import numpy as np

    N = len(SOS)
    EOS_adj = EOS0.copy()

    # precompute absolute first differences
    # note: req_el_1d and t_plot share indices (same length)
    dreq = np.abs(np.diff(req_el_1d))
    # quick helper: turn datetime array to mpl datenums if needed
    # here t_plot is assumed already in Matplotlib date numbers (float)

    for n in range(N):
        # indices within this scan window
        in_scan = (t_plot >= SOS[n]) & (t_plot <= EOS0[n])
        idx = np.where(in_scan)[0]
        if idx.size < (flat_n + 1):
            continue  # not enough samples

        # find last big step within the window (use diffs inside idx range)
        # diffs are between idx[i] and idx[i]+1
        di = idx[:-1]  # valid start indices for dreq
        big = np.where(dreq[di] > step_deg)[0]
        if big.size == 0:
            # no big step-optionally still try to find any flat segment near end
            # (keep EOS0 if not desired)
            continue
        last_big_i = big[-1]   # position in di
        start_j = last_big_i + 1  # look AFTER the last big step

        # scan for first flat run of length flat_n
        # we check max-min over a sliding window
        found = False
        for j in range(start_j, len(idx) - flat_n):
            window = req_el_1d[idx[j : j + flat_n]]
            if (np.max(window) - np.min(window)) <= flat_eps:
                EOS_adj[n] = t_plot[idx[j]]  # first sample of flat run
                found = True
                break

        # enforce EOS < next SOS if exists
        if n < N - 1 and EOS_adj[n] >= SOS[n + 1]:
            # back off slightly to ensure strict ordering
            EOS_adj[n] = np.nextafter(SOS[n + 1], SOS[n + 1] - 1.0)

    return EOS_adj


def adjust_eos_per_antenna1(t_plot, req_el_1d, SOS, EOS0,
                           el_thresh=87.5):
    """
    Returns EOS_adj (len = N scans) for ONE antenna.

    Logic:
      For each scan n, search within [SOS[n], EOS0[n]] for the first
      timestamp where requested elevation exceeds `el_thresh` (default 87.5 deg).
      If found, set EOS_adj[n] to that timestamp; otherwise, keep EOS0[n].
      Ensures EOS_adj[n] < SOS[n+1] (if exists).
    """
    import numpy as np

    N = len(SOS)
    EOS_adj = EOS0.copy()

    for n in range(N):
        # indices within this scan window
        in_scan = (t_plot >= SOS[n]) & (t_plot <= EOS0[n])
        idx = np.where(in_scan)[0]
        if idx.size == 0:
            continue

        # find first time elevation exceeds threshold
        print('max req_el_1d[idx]:', np.max(req_el_1d[idx]))
        over_thresh = np.where(req_el_1d[idx] > el_thresh)[0]
        if over_thresh.size > 0:
            EOS_adj[n] = t_plot[idx[over_thresh[0]]]

        # enforce EOS < next SOS if exists
        if n < N - 1 and EOS_adj[n] >= SOS[n + 1]:
            EOS_adj[n] = np.nextafter(SOS[n + 1], SOS[n + 1] - 1.0)

    return EOS_adj

def adjust_eos_per_antenna(t_plot, req_el_1d, SOS, EOS0,
                           flat_eps=1e-9,   # max abs change between consecutive samples within flat region
                           flat_n=300,      # number of consecutive samples (300 seconds) considered "flat"
                           jump_thresh=0.05):  # min abs change before flat region (deg/s)
    """
    Returns EOS_adj (len = N scans) for ONE antenna.

    Logic:
      For each scan n, within [SOS[n], EOS0[n]]:
        1) Find the first flat run of length flat_n such that
           abs(diff(req_el_1d)) <= flat_eps for all steps.
        2) If found, look backward for the last sample before this flat run
           that shows a large change (dw > jump_thresh).
           This point marks the transition into stow (end of celestial tracking).
           Set EOS_adj[n] to that timestamp.
        3) If no such run is found, keep EOS0[n].
        4) Ensure EOS_adj[n] < SOS[n+1] if exists.
    """
    import numpy as np

    N = len(SOS)
    EOS_adj = EOS0.copy()

    for n in range(N):
        in_scan = (t_plot >= SOS[n]) & (t_plot <= EOS0[n])
        idx = np.where(in_scan)[0]
        if idx.size < flat_n:
            continue

        found = False
        for j in range(0, len(idx) - flat_n + 1):
            w = req_el_1d[idx[j : j + flat_n]]
            dw = np.abs(np.diff(w))
            if np.all(dw <= flat_eps):
                # found start of flat region
                if j > 0:
                    # find last large change before flat region
                    pre_dw = np.abs(np.diff(req_el_1d[idx[:j]]))
                    large_jump = np.where(pre_dw > jump_thresh)[0]
                    if large_jump.size > 0:
                        EOS_adj[n] = t_plot[idx[large_jump[-1]]]  # last large jump
                    else:
                        EOS_adj[n] = t_plot[idx[j]]  # fallback to start of flat region
                else:
                    EOS_adj[n] = t_plot[idx[j]]  # flat starts right away
                found = True
                break

        # enforce EOS < next SOS if exists
        if n < N - 1 and EOS_adj[n] >= SOS[n + 1]:
            EOS_adj[n] = np.nextafter(SOS[n + 1], SOS[n + 1] - 1.0)

    return EOS_adj



def plot_pointing_tracks(trange, savefig=False, outdir="/common/webplots/ant_track",
                         showplt=False, coord='el'):
    """
    Fetch az/el info for given datetime range (UTC) and produce tracking figures.

    Parameters
    ----------
    trange : sequence of datetime
    savefig : bool
    outdir : str
    showplt : bool
    coord : str or iterable of str
        Coordinate keys to plot. Supported keys: 'el', 'az', 'fa'. Provide multiple
        keys to generate several figures from a single SQL query.
    """

    if isinstance(coord, str):
        coord_keys = [coord]
    else:
        coord_keys = list(coord)
    if not coord_keys:
        raise ValueError("coord must include at least one key.")
    coord_keys = [str(c).lower() for c in coord_keys]
    supported_keys = {'el', 'az', 'fa'}
    invalid = [c for c in coord_keys if c not in supported_keys]
    if invalid:
        raise ValueError("Unsupported coord keys {}. Use subset of {}.".format(invalid, sorted(supported_keys)))
    seen = set()
    coord_keys = [c for c in coord_keys if not (c in seen or seen.add(c))]

    coord_config = {
        'el': {
            'requested_key': 'RequestedElevation',
            'actual_key': 'ActualElevation',
            'ylabel': 'Elevation [deg]',
            'special_ylabel': 'Declination [deg]',
            'req_label': 'Req El',
            'act_label': 'Act El',
            'special_req_label': 'Req Dec',
            'special_act_label': 'Act Dec',
            'ylim': (0, 90),
            'title': 'Elevation'
        },
        'az': {
            'requested_key': 'RequestedAzimuth',
            'actual_key': 'ActualAzimuth',
            'ylabel': 'Azimuth [deg]',
            'special_ylabel': 'Hour Angle [deg]',
            'req_label': 'Req Az',
            'act_label': 'Act Az',
            'special_req_label': 'Req HA',
            'special_act_label': 'Act HA',
            'ylim': (0, 360),
            'title': 'Azimuth'
        }
    }

    data_trange = Time(trange)
    azeldict = pc.get_sql_info(data_trange)

    projdict = get_projects(data_trange[0]) or {}
    time = azeldict['Time']
    t_datetime = time.datetime
    t_plot = time.plot_date
    nt, nant = azeldict['ActualElevation'].shape

    SOS = EOS0 = None
    if projdict and 'Timestamp' in projdict:
        timestamp = Time(projdict['Timestamp'], format='lv')
        tidxs = np.where(timestamp <= data_trange[1])[0]
        if len(tidxs) >= 0:
            for k, v in projdict.items():
                projdict[k] = v[tidxs]
        SOS = Time(projdict['Timestamp'], format='lv').plot_date
        SOS = np.asarray(SOS, dtype=float)
        EOS0 = np.roll(SOS, -1)
        EOS0[-1] = data_trange[1].plot_date
        projects = np.asarray(projdict['Project'])
        print(Time(projdict['Timestamp'], format='lv').iso, projects)
    else:
        projects = None

    annotation_eos_by_ant = None
    if projects is not None and SOS is not None:
        az_key = 'RequestedAzimuth'
        if az_key in azeldict:
            annotation_eos_by_ant = np.zeros((len(SOS), nant), dtype=float)
            req_az_full = np.asarray(azeldict[az_key], dtype=float)
            for aidx in range(nant):
                req_series = req_az_full[:, aidx]
                annotation_eos_by_ant[:, aidx] = adjust_eos_per_antenna(
                    t_plot, req_series, SOS, EOS0)

    ant16_idx = 15 if nant > 15 else (nant - 1)
    figures = []

    save_dir = outdir
    datestamp = None
    if savefig:
        if trange[1].hour < 7:
            datelocal = trange[1].date() - timedelta(days=1)
        else:
            datelocal = trange[1].date()
        if datelocal.year < datetime.now().year:
            save_dir = os.path.join(outdir, datelocal.strftime('%Y'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        datestamp = datelocal.strftime('%Y%m%d')

    def _save_plot(fig, coord_tag):
        if not savefig:
            return
        figname = "{}/track_{}_{}.jpg".format(save_dir, coord_tag, datestamp)
        fig.savefig(figname, dpi=100, bbox_inches='tight')
        print("Wrote", figname)
        figfiles = sorted(glob('{}/track_{}_20*'.format(save_dir, coord_tag)))
        if figfiles:
            shutil.copy(figfiles[-1], '{}/track_{}_latest.jpg'.format(save_dir, coord_tag))

    for coord_key in coord_keys:
        if coord_key in ('el', 'az'):
            cfg = coord_config[coord_key]
            fig, axs = plt.subplots(figsize=(18, 9), ncols=4, nrows=4, sharex=False, sharey=False)
            figures.append(fig)
            axs = axs.flatten()
            ymin, ymax = cfg['ylim']
            for aidx in range(nant):
                ax = axs[aidx]
                ax.cla()
                ymin_cur, ymax_cur = ymin, ymax
                if coord_key == 'az' and aidx == ant16_idx:
                    ymin_cur, ymax_cur = -90, 90
                ax.set_ylim(ymin_cur, ymax_cur)
                if aidx % 4 == 0:
                    ax.set_ylabel(cfg['ylabel'])
                if aidx >= 12:
                    ax.set_xlabel('Time [UT]')
                if aidx == ant16_idx:
                    ax.set_ylabel(cfg['special_ylabel'])
                if projects is not None and SOS is not None and annotation_eos_by_ant is not None:
                    eos_adj = annotation_eos_by_ant[:, aidx]
                    gm = np.zeros_like(t_plot, dtype=bool)
                    for n in range(len(SOS)):
                        left = SOS[n]
                        right = eos_adj[n]
                        gm |= (t_plot >= left) & (t_plot <= right)
                    req_data = azeldict[cfg['requested_key']][:, aidx].astype(float)
                    act_data = azeldict[cfg['actual_key']][:, aidx].astype(float)
                    req = ma.masked_array(req_data, mask=~gm)
                    act = ma.masked_array(act_data, mask=~gm)
                    annotate_observing_intervals(ax, SOS, eos_adj, projects)
                if aidx == ant16_idx:
                    ax.plot(t_datetime, req, label=cfg['special_req_label'], linestyle='-', c='#7f7f7f', lw=5)
                    ax.plot(t_datetime, act, label=cfg['special_act_label'], linestyle='none', c='C1', marker='o', markersize=1)
                else:
                    ax.plot(t_datetime, req, label=cfg['req_label'], linestyle='-', c='#7f7f7f', lw=5)
                    ax.plot(t_datetime, act, label=cfg['act_label'], linestyle='none', c='C1', marker='o', markersize=1)
                ax.set_title('Ant {}'.format(aidx + 1))
                ax.grid(False)
                ax.set_ylim(ymin_cur, ymax_cur)
                time_fmt = DateFormatter("%H:%M")
                ax.xaxis.set_major_formatter(time_fmt)
                if aidx >= nant - 2:
                    ax.legend(loc='lower right')
            fig.autofmt_xdate()
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.suptitle('Antenna {} Tracking Monitor for {}'.format(cfg['title'], trange[0].strftime('%Y-%m-%d')), y=0.995)
            _save_plot(fig, coord_key)
        elif coord_key == 'fa':
            if 'ParallacticAngle' not in azeldict:
                raise KeyError("Azeldict missing 'ParallacticAngle' required for 'fa' plot.")
            required_keys = ['AntA_Feed_Angle', 'AntA_Feed_Offset', 'AntA_Feed_Error']
            missing = [k for k in required_keys if k not in azeldict]
            if missing:
                raise KeyError("Azeldict missing keys {} required for 'fa' plot.".format(missing))
            fig, axs = plt.subplots(figsize=(18, 9), ncols=4, nrows=4, sharex=False, sharey=False)
            figures.append(fig)
            axs = axs.flatten()
            feed_angle = -np.asarray(azeldict['AntA_Feed_Angle'], dtype=float)
            feed_offset = np.asarray(azeldict['AntA_Feed_Offset'], dtype=float)
            feed_error = np.asarray(azeldict['AntA_Feed_Error'], dtype=float)
            par_angle = np.asarray(azeldict['ParallacticAngle'], dtype=float)
            t_Anta_plt = azeldict['AntA_Time'].datetime
            for aidx in range(nant):
                ax = axs[aidx]
                ax.cla()
                ax.set_ylim(-180, 180)
                if aidx % 4 == 0:
                    ax.set_ylabel('Angle [deg]')
                if aidx >= 12:
                    ax.set_xlabel('Time [UT]')
                ax.set_title('Ant {}'.format(aidx + 1))
                if projects is not None and SOS is not None and annotation_eos_by_ant is not None:
                    annotate_observing_intervals(ax, SOS, annotation_eos_by_ant[:, aidx], projects)
                else:
                    req = azeldict[cfg['requested_key']][:, aidx]
                    act = azeldict[cfg['actual_key']][:, aidx]
                if aidx == ant16_idx:
                    ax.plot(t_Anta_plt, feed_offset, label='Pos Off', linestyle='-', c='#7f7f7f', lw=5)
                    ax.plot(t_Anta_plt, feed_error, label='Pos Err', linestyle='-', c="#dadada", lw=5)
                    ax.plot(t_Anta_plt, feed_angle, label='Pos Angle', linestyle='none', c='C1', marker='o', markersize=1)
                else:
                    ax.plot(t_datetime, par_angle[:, aidx], label='Par Ang', linestyle='none', c='C1', marker='o', markersize=1)
                ax.grid(False)
                time_fmt = DateFormatter("%H:%M")
                ax.xaxis.set_major_formatter(time_fmt)
                if aidx == ant16_idx or aidx >= nant - 2:
                    ax.legend(loc='lower right')
            fig.autofmt_xdate()
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.suptitle('Antenna Feed Angle Monitor for {}'.format(trange[0].strftime('%Y-%m-%d')), y=0.995)
            _save_plot(fig, coord_key)

    if showplt:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=("Plot Requested vs Actual/Feed pointing per antenna "
                     "for 12:00-04:00 UT window. "
                     "If --date omitted, uses yesterday (UTC) as base date."))
    parser.add_argument('--date', type=str, default=None,
                        help='Base date in YYYY-MM-DD (UTC)')
    parser.add_argument('--showplt', action='store_true', help='show plot instead of saving')
    parser.add_argument('--outdir', type=str, default='/common/webplots/ant_track', help='Output dir when using --save.')
    parser.add_argument('--coord', nargs='+', default=['el', 'az', 'fa'], choices=['el', 'az', 'fa'],
                        help="Coordinate keys to plot (default: el az). Include 'fa' for feed angle.")
    args = parser.parse_args()

    trange_dt = build_timerange_daily(args.date)
    print("Using time range (UTC):", trange_dt[0], "to", trange_dt[1])
    plot_pointing_tracks(trange_dt, savefig=True, outdir=args.outdir, showplt=args.showplt, coord=args.coord)

if __name__ == '__main__':
    main()


'''
Example usage:
$ python /common/python/current/daily_track.py --date 2025-07-24 --coord fa
or in ipython session:
from daily_track import plot_pointing_tracks
from datetime import datetime
trange = [datetime(2025, 9, 9, 12, 50, 0), datetime(2025, 9, 9, 14, 50, 0)]
plot_pointing_tracks(trange, savefig=True, outdir='/common/webplots/ant_track/', showplt=False, coord=['fa'])

'''
# # Define fixed time range for plotting
# trange = [
#     datetime(2025, 10, 6, 12, 0, 0),
#     datetime(2025, 10, 7, 4, 0, 0)
# ]
#
#
# trange = [
#     datetime(2025, 8, 5, 12, 0, 0),
#     datetime(2025, 8, 11, 4, 0, 0)
# ]
