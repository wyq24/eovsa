import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
# from astropy.time import Time
import pipeline_cal as pc
from util import Time
from glob import glob
import shutil


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


def plot_elevation_tracks(trange, savefig=False, outdir="/common/webplots/ant_track", showplt=False):
    """
    Fetch az/el info for given datetime range (UTC) and plot
    Requested vs Actual Elevation for each antenna in a 4x4 grid.
    Last axis left empty for common legend.
    """

    # Example: load data for the desired time window
    # t = Time.now()
    data_trange = Time(trange)
    azeldict = pc.get_sql_info(data_trange)

    # Extract time and flag
    time = azeldict['Time']
    trackflag = azeldict['TrackFlag']

    # Convert to datetime for matplotlib
    t_datetime = time.datetime

    # plt.show()
    # azeldict.keys()
    # dict_keys(['dElevation', 'ActualAzimuth', 'RFSwitch', 'TrackFlag', 'RequestedElevation', 'dAzimuth', 'Receiver', 'ParallacticAngle', 'RequestedAzimuth', 'Time', 'ActualElevation', 'TrackSrcFlag', 'LF_Rcvr'])

    nt, nant = azeldict['ActualAzimuth'].shape

    fig, axs = plt.subplots(figsize=(10, 8), ncols=4, nrows=4, sharex=True, sharey=True)
    axs = axs.flatten()
    for aidx in range(nant-1):
        ax = axs[aidx]
        ax.cla()
        ax.plot(t_datetime, azeldict['RequestedElevation'][:, aidx], label='Req El', linestyle='-',c='C0', lw=4)
        ax.plot(t_datetime, azeldict['ActualElevation'][:, aidx], label='Act El', linestyle='none',c='C1', marker='o', markersize=1)
        ax.set_title('Ant {}'.format(aidx + 1))
        ax.grid(True)
        if aidx % 4 == 0:
            ax.set_ylabel('Elevation [deg]')
        if aidx >= 12:
            ax.set_xlabel('Time [UT]')

    ax = axs[-1]
    ax.plot([], [], label='Req El', linestyle='-',c='C0', lw=4)
    ax.plot([], [], label='Act El', linestyle='none',c='C1',
            marker='o', markersize=1)
    ax.axis('off')
    ax.legend(loc='center')
    ax.set_ylim(0,90)
    time_fmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(time_fmt)
    # fig.suptitle('Requested vs Actual Elevation by Antenna')
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('Antenna Tracking Monitor for {}'.format(trange[0].strftime('%Y-%m-%d')), y=0.995)
    if savefig:
        if trange[1].hour < 7:
            # Data time is earlier than 7 UT (i.e. on previous local day) so
            # use previous date at 20 UT.
            datelocal = trange[1].date() - timedelta(days=1)
        else:
            datelocal = trange[1].date()
        figname = '{}/track_{}.jpg'.format(outdir, datelocal.strftime('%Y%m%d'))
        fig.savefig(figname, dpi=150)
        figfiles = np.sort(glob('{}/track_20*'.format(outdir)))
        shutil.copy(figfiles[-1], '{}/track_latest.jpg'.format(outdir))

    if showplt:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description=("Plot Requested vs Actual Elevation per antenna "
                     "for 11:00-05:00 UT window. "
                     "If --date omitted, uses yesterday (UTC) as base date.")
    )
    parser.add_argument('--date', type=str, default=None,
                        help='Base date in YYYY-MM-DD (UTC)')
    parser.add_argument('--showplt', action='store_true', help='show plot instead of saving')
    parser.add_argument('--outdir', type=str, default='/common/webplots/ant_track', help='Output dir when using --save.')
    args = parser.parse_args()

    trange_dt = build_timerange_daily(args.date)
    print("Using time range (UTC):", trange_dt[0], "to", trange_dt[1])
    plot_elevation_tracks(trange_dt, savefig=True, outdir=args.outdir, showplt=args.showplt)

if __name__ == '__main__':
    main()


# # Define fixed time range for plotting
# trange = [
#     datetime(2025, 10, 6, 12, 0, 0),
#     datetime(2025, 10, 7, 4, 0, 0)
# ]
#
#
# trange = [
#     datetime(2025, 10, 7, 12, 0, 0),
#     datetime(2025, 10, 8, 4, 0, 0)
# ]

# # Create figure and axis
# fig, ax = plt.subplots(figsize=(10, 4))
#
# # Plot selected antennas
# for ant in [0, 3, 4]:
#     ax.plot(t_datetime, trackflag[:, ant].astype(int), label='Ant {}'.format(ant + 1))
#
# # Axis labels, title, grid, legend
# ax.set_xlabel('Time (UTC)')
# ax.set_ylabel('Track Flag (0 = off, 1 = tracking)')
# ax.set_title('Track Flag vs Time')
# ax.grid(True)
# ax.legend()
# fig.tight_layout()
#
# # Use the specified datetime range for x-axis
# ax.set_xlim(trange)
#
# # Optionally format x-ticks nicely
# fig.autofmt_xdate()
#