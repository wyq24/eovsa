# Purpose: Contains a bunch of routines for finding and plotting when
#          calibration sources and the Sun are up.  Also contains the
#          key routine make_sched(), which creates a solar schedule
#          for a given date.
# History:
#  2018-09-16  DG
#    First wrote whenup(), based on whatup.py
#  2018-09-18  DG
#    Completed the routines whenup(), sunup(), plot_sun(), and make_sched()
#  2019-01-18  DG
#    A bug occurred on some dates due to source not being within 0.1 degree of
#    10 degrees altitude.  Introduced a for loop and test in both whenup() and
#    make_sched() to use a wider window.
#  2019-05-20  DG
#    Added a 1-min PHASECAL with sequence solar.fsq just before the SKYCALTEST.
#  2020-10-01  DG
#    Added a check for day numbers 259-287, when 3C273 (1229+020) is too close
#    to the Sun.  On those dates, replace any 1229+020 lines with 1331+305.
#  2020-10-29  DG
#    Comment out a bunch of lines due to 27-m not working--lines are commented
#    out using #**.  Note that two lines were added that have to be removed,
#    and they also have #** in a comment on those lines.
#  2020-10-31  DG
#    Looks like the 27-m is back, so I reverted the code back to the original.
#  2021-09-09  DG
#    The schedule broke today because it is day 251, which is a transition between
#    three and two calibrations during the day.  I changed the iday range from >251
#    to >=251, and now it seems to work.
#  2021-09-22  DG
#    My replacement of 1229+020 by 1331+305 for day numbers 259-287 was broken, 
#    because 1331+305 was not up yet!  I now add an appropriate number of minutes 
#    to the times of those lines.
#  2022-03-14  DG
#    Changes to reduce the length of PHASECALs to from 25-30 minutes to 20 minutes.
#  2022-05-14  DG
#    Added a remove_cal() function to remove all 27-m calibration lines from
#    the solar schedule created by make_sched().
#  2023-05-06  DG
#    Temporarily shift the last calibration of the day 1 hour earlier so as not to
#    overlap with SRH solar observations (for summer 2023 at least).
#  2023-10-12  DG
#    Remove solpnt.trj filename from each of the SOLPNTCAL lines, since with the
#    new feeds the search pattern is different for the two types of antenna.  This is
#    now set in the SOLPNTCAL.ctl file itself.
#  2023-10-16  DG
#    Added 2 minutes to SOLPNTCAL to account for longer trajectory duration.
#  2023-10-29  DG
#    Removed the now unneeded SKYCALTEST
#  2024-01-30  DG
#    Fixed remove_cal(), which was broken due to removal of SKYCALTEST.
#  2024-04-02  DG
#    Add new GAINSOLPNT calibration lines right after the SOLPNTCALs.  Will later
#    remove the now-redundant GAINCALTEST scans...
#  2024-04-30  DG
#    Finally removed the now-redundant GAINCALTEST scans.
#  2024-05-25  DG
#    Adjust sunrise for dates between May 10 and July 31 by 10 min to account for
#    shadowing by Black Mt.
#  2024-05-30  DG
#    Change end STOW line to REWIND.
#  2026-07-20  SY
#    Add Ant 13 FEM power events to automatically generated solar schedules.
#  2026-08-21  SY
#    Power down Ant 13 during sunset-to-calibration gaps longer than 10 minutes.
#  2026-08-31  SY
#    Work around the Ant 3 elevation soft-limit trips at low Sun elevations:
#    the day's first SUN scan now uses SUN_NO_ANT3 (Ant 3 left at stow), a
#    SUN_ANT3 line joins Ant 3 to the solar track once the Sun rises above
#    ANT3_SUN_LIMIT_DEG, and a STOW_ANT3 line parks Ant 3 in the evening
#    when the Sun sets below the same limit.  Also taught remove_cal() to
#    keep STOW_ANT3 lines (and the array-wide STOW they displace) and to
#    deduplicate its kept lines.  Companion change in schedule.py classifies
#    SUN_NO_ANT3/SUN_ANT3 scans as normal solar observing.
#

import os
from util import Time
import eovsa_cat
from eovsa_visibility import *
import numpy as np
from astropy.time import TimeDelta

def deg(rad):
    return (rad * 180./np.pi + 180) % 360 - 180

def deg2(rad):
    return rad * 180./np.pi

# Minimum Sun elevation (deg) at which Ant 3 may track the Sun.  Ant 3's
# pointing model currently has P7 = -5.1013 deg (elevation collimation /
# encoder-zero term), so its elevation demand runs ~5.1 deg below the other
# antennas and reaches the 10-deg drive soft limit while the Sun is still
# at ~15.1 deg.  10 (soft limit) + 5.1 (P7 offset) + 0.5 (margin) = 15.6.
# If Ant 3's pointing model is recalibrated, update this to
# 10 - P7 + 0.5 margin.  See ovsa-ops-memo docs/reference/ant3-sun-late-start.md.
ANT3_SUN_LIMIT_DEG = 15.6

def whenup(date=None,verbose=False):
    ''' Find out the times when preferred sources are up for a given date. 
        Prints the result of source rise and set times for preferred sources:
        Sun, 0319+415, 1229+020, 1331+305, 2136+006, 2253+161
        
        Optional arguments:
           date: Time() object giving start date (time of day is ignored) 
                 or None for current date.
           verbose: If True, actually print the table of times to the screen
           
        Returns: dictionary of time objects with keys 'taz', 'teq', 'tgap', each
                 being a two-element Time() object, where first time is
                 start (rise) and second is end (set), and 'lines', which is a
                 printable table of times.
    '''
    srclist = ['Sun     ', '0319+415', '1229+020', '1331+305', '2136+006', '2253+161']

    # Use the 24-h day specified by date (i.e. drop the time of day)
    if date is None:
        # Use today's date
        mjd = int(Time.now().mjd)
    else:
        mjd = int(date.mjd)
    t = Time(mjd,format='mjd')

    # Add times in 1-min steps up to duration dur (hours)
    ts = t + TimeDelta(np.arange(0.,24.,1./60.)/24.,format='jd')

    aa = eovsa_cat.eovsa_array_with_cat(include_satellites=False)

    nt = len(ts)
    ra = np.zeros((len(srclist),nt))
    ha = np.zeros((len(srclist),nt))
    dec = np.zeros((len(srclist),nt))
    alt = np.zeros((len(srclist),nt))
    az = np.zeros((len(srclist),nt))
    taz = []
    teq = []
    tgap = []
    lines = []
    for i in range(nt):
        aa.set_jultime(ts[i].jd)    
        lst = aa.sidereal_time()

        for j,srcname in enumerate(srclist):
            src = aa.cat[srcname.split()[0]]
            src.compute(aa)
            ra[j,i] = src.ra
            ha[j,i] = lst - src.ra
            dec[j,i] = src.dec
            alt[j,i] = src.alt
            az[j,i] = src.az

    ra_deg = deg(ra)
    ha_deg = deg(ha)
    dec_deg = deg(dec)
    alt = deg(alt)
    az_deg = deg2(az)
    lines.append('Source     Alt-Az    Equatorial     Gap')
    lines.append(' Name    Rise   Set  Rise   Set  Start  End')
    lines.append('-------- ----- ----- ----- ----- ----- -----')
    for j in range(len(srclist)):
        iset_eq = np.where(np.abs(ha_deg[j] - 55.0) < 0.15)[0][0]
        irise_eq = np.where(np.abs(ha_deg[j] + 55.0) < 0.15)[0][0]
        trise_eq = ts[irise_eq]
        tset_eq = ts[iset_eq]
        if iset_eq < irise_eq:
            if j == 0:
                tset_eq += TimeDelta(1,format='jd')
            else:
                tset_eq += TimeDelta(1436./60./24.,format='jd')
        for iwindow in np.arange(0.1,0.2,0.01):
            try:
                i1 = np.where(np.abs(alt[j] - 10.0) < iwindow)[0][0]
                i2 = np.where(np.abs(alt[j] - 10.0) < iwindow)[0][1]
                if i2 == i1+1:
                    i2 = np.where(np.abs(alt[j] - 10.0) < iwindow)[0][2]
                break
            except:
                print 'Window',iwindow,'did not work.  Trying again'
        if alt[j,i1] < alt[j,i1+1]:
            irise_az = i1
            iset_az = i2
        else:
            irise_az = i2
            iset_az = i1
        trise_az = ts[irise_az]
        tset_az = ts[iset_az]
        if iset_az < irise_az:
            if j == 0:
                tset_az += TimeDelta(1,format='jd')
            else:
                tset_az += TimeDelta(1436./60./24.,format='jd')
        if j == 1:
            up, = np.where(alt[j] > 10.)
            j1 = np.where(np.abs(az_deg[j,up] - 35.0) < 1.0)[0][0]
            j2 = np.where(np.abs(az_deg[j,up] - 325.0) < 1.0)[0][0]
            jset = up[j1]
            jrise = up[j2]
            tset_gap = ts[jset]
            trise_gap = ts[jrise]
            if jrise < jset:
                trise_gap += TimeDelta(1436./60./24.,format='jd')
            taz.append(Time([trise_az.iso,tset_az.iso]))
            teq.append(Time([trise_eq.iso,tset_eq.iso]))
            tgap.append(Time([tset_gap.iso,trise_gap.iso]))
            lines.append('{0} {1} {2} {3} {4} {5} {6}'.format(srclist[j], trise_az.iso[11:16], tset_az.iso[11:16], trise_eq.iso[11:16], tset_eq.iso[11:16], tset_gap.iso[11:16], trise_gap.iso[11:16]))
        else:
            taz.append(Time([trise_az.iso,tset_az.iso]))
            teq.append(Time([trise_eq.iso,tset_eq.iso]))
            tgap.append((None,None))
            lines.append('{0} {1} {2} {3} {4} {5} {6}'.format(srclist[j], trise_az.iso[11:16], tset_az.iso[11:16], trise_eq.iso[11:16], tset_eq.iso[11:16], ' --- ',' --- '))
    if verbose:
        for line in lines:
            print line
    return {'source':srclist,'taz':taz, 'teq':teq, 'tgap':tgap, 'lines':lines}

def sunup(daterange):
    ''' Find out the times when the Sun is up for a given date range. 
           
        Returns: dictionary of time objects with keys 'taz', 'teq', each
                 being a two-element Time() object, where first time is
                 start (rise) and second is end (set)
    '''
    # Use the 24-h day specified by daterange (i.e. drop the time of day)
    mjd1 = int(daterange[0].mjd)
    mjd2 = int(daterange[1].mjd)
    # sunup() only needs the Sun, which is already present in the base array
    # catalog.  Avoid loading calibrators and satellite TLE files here.
    aa = eovsa_cat.eovsa_array()
    taz_rise = []
    teq_rise = []
    taz_set = []
    teq_set = []
    mjd_list = []
    for mjd in range(mjd1,mjd2+1):
        t = Time(mjd,format='mjd')
        # Add times in 1-min steps up to duration dur (hours)
        ts = t + TimeDelta(np.arange(0.,24.,1./60.)/24.,format='jd')

        nt = len(ts)
        ha = np.zeros(nt)
        alt = np.zeros(nt)
        az = np.zeros(nt)
        lines = []
        for i in range(nt):
            aa.set_jultime(ts[i].jd)    
            lst = aa.sidereal_time()

            src = aa.cat['Sun']
            src.compute(aa)
            ha[i] = lst - src.ra
            alt[i] = src.alt
            az[i] = src.az

        ha_deg = deg(ha)
        alt = deg(alt)
        az_deg = deg2(az)
        iset_eq = np.where(np.abs(ha_deg - 55.0) < 0.15)[0][0]
        irise_eq = np.where(np.abs(ha_deg + 55.0) < 0.15)[0][0]
        trise_eq = ts[irise_eq]
        tset_eq = ts[iset_eq]
        if iset_eq < irise_eq:
            tset_eq += TimeDelta(1,format='jd')
        for iwindow in np.arange(0.1,0.2,0.01):
            try:
                idx, = np.where(np.abs(alt - 10.0) < iwindow)  #****Was 0.1
                i1 = idx[0]
                i2 = idx[1]
                if i2 == i1+1:
                    if len(idx) > 2:
                        i2 = idx[2]
                    else:
                        i2 = 1439  # Transition is at end
                break
            except:
                print 'Window',iwindow,'did not work.  Trying again'
        if alt[i1] < alt[i1+1]:
            irise_az = i1
            iset_az = i2
        else:
            irise_az = i2
            iset_az = i1
        trise_az = ts[irise_az]
        tset_az = ts[iset_az]
        if iset_az < irise_az:
            tset_az += TimeDelta(1,format='jd')
        taz_rise.append(trise_az.mjd)
        taz_set.append(tset_az.mjd)
        teq_rise.append(trise_eq.mjd)
        teq_set.append(tset_eq.mjd)
        mjd_list.append(mjd)
    return {'date':Time(mjd_list,format='mjd'),
            'taz_rise':Time(taz_rise,format='mjd'),'teq_rise':Time(teq_rise,format='mjd'),
             'taz_set':Time(taz_set, format='mjd'), 'teq_set':Time(teq_set, format='mjd')}

def sun_limit_times(t, limit_deg):
    ''' Find the times when the Sun rises above and sets below a given
        elevation on the day specified by t.

        :param t: Date for the calculation (time of day is ignored).
        :type t: astropy.time.Time
        :param limit_deg: Elevation threshold in degrees.
        :type limit_deg: float
        :returns: Tuple (trise, tset) of Time objects at 1-min resolution,
                  rounded to the safe side: trise is the first minute at or
                  above limit_deg on the rising branch, tset the last minute
                  at or above limit_deg before the following setting branch
                  (tset can fall after 24 UT, i.e. on the next MJD).  Either
                  element is None if no such crossing is found.
        :rtype: tuple
    '''
    mjd = int(t.mjd)
    # Add times in 1-min steps for 48 h, to catch summer settings after 0 UT
    # on the next MJD
    ts = Time(mjd,format='mjd') + TimeDelta(np.arange(0.,48.,1./60.)/24.,format='jd')

    aa = eovsa_cat.eovsa_array()
    nt = len(ts)
    alt = np.zeros(nt)
    for i in range(nt):
        aa.set_jultime(ts[i].jd)
        src = aa.cat['Sun']
        src.compute(aa)
        alt[i] = src.alt

    alt = deg2(alt)
    above = alt >= limit_deg
    up, = np.where(np.logical_and(np.logical_not(above[:-1]), above[1:]))
    dn, = np.where(np.logical_and(above[:-1], np.logical_not(above[1:])))
    if len(up) == 0:
        return None, None
    irise = up[0] + 1          # first minute at or above the limit
    dn = dn[dn >= irise]       # setting crossing paired with that rise
    if len(dn) == 0:
        return ts[irise], None
    return ts[irise], ts[dn[0]]   # dn[0] = last minute at or above the limit


def plot_sun(sun):
    ''' Plot the output of sunup() in a nice format for visualizing the
        relevant times.
        
        Returns:
           ax    The axis of the plot, for potential use by make_sched for overplotting
    '''
    import matplotlib.pylab as plt
    t = np.array([0.0, 0.141, 0.1639, 0.3049, 0.3805, 0.6847, 0.7278, 0.7625, 0.816, 1.1188])
    mjd0 = sun['date'][0].mjd
    mjd1 = sun['date'][-1].mjd
    y = np.array([mjd0, mjd0, mjd1, mjd1])
    out = whenup(sun['date'][0])
    t0 = out['teq'][1][0].mjd % 1
    f, ax = plt.subplots(1,1)
    ax.plot(sun['taz_rise'].mjd-sun['date'].mjd,sun['date'].mjd,'--',color='C1')
    ax.plot(sun['taz_set'].mjd-sun['date'].mjd,sun['date'].mjd,'--',color='C1')
    ax.plot(sun['taz_rise'].mjd-sun['date'].mjd-0.0583,sun['date'].mjd,':',color='C1')
    ax.plot(sun['taz_set'].mjd-sun['date'].mjd+0.0583,sun['date'].mjd,':',color='C1')
    ax.plot(sun['teq_rise'].mjd-sun['date'].mjd,sun['date'].mjd,'-',color='C0')
    ax.plot(sun['teq_set'].mjd-sun['date'].mjd,sun['date'].mjd,'-',color='C0')
    colors = ['C3','C3','C4','C5','C6','C7']
    for k,i in enumerate([0,2,4,5,7,8]):
        x = np.array([t0 + t[i], t0+t[i+1], (mjd0-mjd1)*0.0027378+t0+t[i+1],(mjd0-mjd1)*0.0027378+t0+t[i]])
        ax.fill(x,y,color=colors[k],alpha=0.25)
        ax.fill(x+1,y,color=colors[k],alpha=0.25)
        ax.fill(x-1,y,color=colors[k],alpha=0.25)
    ax.plot(np.ones(2)*0.7708,[mjd0,mjd1],color='black') # 18:30
    ax.plot(np.ones(2)*0.8958,[mjd0,mjd1],color='black') # 21:30
    ax.plot(np.ones(2)*0.6319,[mjd0,mjd1],color='gray') # 15:10
    ax.plot(np.ones(2)*0.6563,[mjd0,mjd1],color='gray') # 15:45
    ax.plot(np.ones(2)*0.7153,[mjd0,mjd1],color='gray') # 17:10
    ax.plot(np.ones(2)*0.7396,[mjd0,mjd1],color='gray') # 17:45
    ax.plot(np.ones(2)*0.8264,[mjd0,mjd1],color='gray') # 19:50
    ax.plot(np.ones(2)*0.8507,[mjd0,mjd1],color='gray') # 20:25
    ax.plot(np.ones(2)*0.9236,[mjd0,mjd1],color='gray') # 22:10
    ax.plot(np.ones(2)*0.9479,[mjd0,mjd1],color='gray') # 22:45
    ax.plot(np.ones(2)*1.0069,[mjd0,mjd1],color='gray') # 00:10
    ax.plot(np.ones(2)*1.0313,[mjd0,mjd1],color='gray') # 00:45
    ax.set_xlim(0.0,1.5)
    ax.set_title(sun['date'][0].iso[:10]+' to '+sun['date'][-1].iso[:10])
    ax.set_ylabel('MJD')
    ax.set_xlabel('Time [in fraction of a day]')
    return ax


def add_ant13_fem_power_events(lines):
    '''Add idempotent Ant 13 FEM power events to a standard solar schedule.

    :param lines: Schedule lines containing an end-of-day ``REWIND``.
    :type lines: list(str)
    :returns: A new list with ``FEMPOWERON`` five minutes before the first
              ``ACQUIRE`` (or first remaining event), an optional idle-gap
              power cycle around the evening calibration, and
              ``FEMPOWEROFF`` after the final calibration.
    :rtype: list(str)

    The helper is used only by the automatic solar-schedule path.  Custom and
    overnight schedules are not decorated.  Existing power events are removed
    first so repeated calls do not duplicate them.
    '''
    clean_lines = []
    for line in lines:
        tokens = line[20:].split()
        command = tokens[0].upper() if tokens else ''
        if command not in ('FEMPOWERON', 'FEMPOWEROFF'):
            clean_lines.append(line)

    def command(line):
        tokens = line[20:].split()
        return tokens[0].upper() if tokens else ''

    acquire_indices = [i for i, line in enumerate(clean_lines)
                       if command(line) == 'ACQUIRE']
    first_acquire_idx = acquire_indices[0] if acquire_indices else None
    last_acquire_idx = acquire_indices[-1] if acquire_indices else None
    rewind_indices = [i for i, line in enumerate(clean_lines)
                      if command(line) == 'REWIND']
    if not rewind_indices:
        return clean_lines
    rewind_idx = rewind_indices[-1]

    # Only a STOW immediately before the evening calibration marks the end of
    # the final solar scan.  An earlier STOW can belong to a separate morning
    # schedule segment and must not trigger an idle-gap cycle.
    stow_idx = None
    if (last_acquire_idx is not None and last_acquire_idx > 0 and
            command(clean_lines[last_acquire_idx - 1]) == 'STOW'):
        stow_idx = last_acquire_idx - 1
    elif last_acquire_idx is None:
        # No-27m schedules have no evening ACQUIRE.  Their retained final
        # STOW is the solar-end marker used for the terminal FEM shutdown.
        for i in range(rewind_idx - 1, -1, -1):
            if command(clean_lines[i]) == 'STOW':
                stow_idx = i
                break

    # The first power-on remains the normal five-minute warm-up.  No-27m
    # schedules have no ACQUIRE lines, so retain the old fallback to their
    # first scheduled event.
    initial_idx = first_acquire_idx
    if initial_idx is None:
        for i, line in enumerate(clean_lines):
            if command(line) and command(line) != 'REWIND':
                initial_idx = i
                break
    if initial_idx is None:
        return clean_lines

    def power_line(mjd, macro):
        return (Time(mjd, format='mjd').iso[:19] + ' ' + macro)

    insertions = [(initial_idx, 0,
                   power_line(Time(clean_lines[initial_idx][:19]).mjd -
                              5./1440., 'FEMPOWERON'))]

    # With a long sunset-to-calibration gap, shut down directly after STOW and
    # restart five minutes before the evening (last) ACQUIRE.  The strict
    # comparison intentionally leaves a ten-minute gap powered continuously.
    if stow_idx is not None:
        stow_mjd = Time(clean_lines[stow_idx][:19]).mjd
        if last_acquire_idx is None:
            # No-27m schedules have no evening calibration to restart for.
            insertions.append((stow_idx + 1, 0,
                               power_line(stow_mjd, 'FEMPOWEROFF')))
        else:
            acquire_mjd = Time(clean_lines[last_acquire_idx][:19]).mjd
            gap_seconds = int(round((acquire_mjd - stow_mjd) * 86400.))
            if gap_seconds > 10 * 60:
                insertions.append((stow_idx + 1, 0,
                                   power_line(stow_mjd, 'FEMPOWEROFF')))
                if first_acquire_idx != last_acquire_idx:
                    insertions.append((last_acquire_idx, 1,
                                       power_line(acquire_mjd - 5./1440.,
                                                  'FEMPOWERON')))

    # Preserve the existing final shutdown one minute before REWIND whenever
    # an evening ACQUIRE exists.  For a no-27m schedule with a retained STOW,
    # the STOW shutdown above is the final FEM event.  If no STOW exists, keep
    # the original REWIND-relative fallback instead.
    if last_acquire_idx is not None or stow_idx is None:
        rewind_mjd = Time(clean_lines[rewind_idx][:19]).mjd
        insertions.append((rewind_idx, 0,
                           power_line(rewind_mjd - 1./1440.,
                                      'FEMPOWEROFF')))

    # Insert against original indices from right to left, preserving the
    # explicit STOW -> FEMPOWEROFF order for equal timestamps.
    powered_lines = list(clean_lines)
    for index, order, line in sorted(insertions,
                                     key=lambda item: (item[0], item[1]),
                                     reverse=True):
        powered_lines.insert(index, line)
    return powered_lines

def make_sched(sun=None, t=None, ax=None, verbose=False,
               ant13_fem_power=False, ant3_late=True, ant3_sun_limit=None):
    '''Create a daily solar schedule for a specified date.

    :param sun: A dictionary returned by :func:`sunup` whose date range
                contains ``t``.  When omitted, it is calculated internally.
    :type sun: dict or None
    :param t: Date for the schedule; defaults to the current date.
    :type t: astropy.time.Time or None
    :param ax: Optional plot axis on which to show calibrator ranges.
    :type ax: matplotlib.axes.Axes or None
    :param verbose: Print generated schedule lines when ``True``.
    :type verbose: bool
    :param ant13_fem_power: Decorate the schedule with Ant 13 FEM power
                            events.  This defaults to ``False`` so other
                            callers do not acquire shared hardware ownership.
    :type ant13_fem_power: bool
    :param ant3_late: When True (default), the first SUN scan of the day
                      excludes Ant 3 (macro SUN_NO_ANT3), a SUN_ANT3 line is
                      added once the Sun rises above ant3_sun_limit, and a
                      STOW_ANT3 line is added when the Sun sets below it.
    :type ant3_late: bool
    :param ant3_sun_limit: Sun elevation threshold in degrees for Ant 3
                           tracking; defaults to ANT3_SUN_LIMIT_DEG.
    :type ant3_sun_limit: float or None
    :returns: Text lines representing the generated schedule.
    :rtype: list(str)
    '''
    # From give time, get mjd0 = MJD of Jan 1 for that year
    if t is None:
        t = Time.now()
    imjd = int(t.mjd)
    year = t.iso[:4]
    mjd0 = int(Time(year+'-01-01').mjd)
    # If no sun dictionary is give, create a 2-day one
    if sun is None:
        sun = sunup(Time([t.mjd,t.mjd+1],format='mjd'))
    if ant3_sun_limit is None:
        ant3_sun_limit = ANT3_SUN_LIMIT_DEG
    # Calibration durations, minutes
    refdur = 84.
    caldur = 20.  #35.
    # Get calibrator reference time.  Rather than calculating exact timing for
    # calibrators for each day, we just get it at one reference time and then
    # calculate it for additional days by subtracting 0.0027387*dday
    out = whenup(sun['date'][0])
    t0 = out['teq'][1][0].mjd % 1   # Reference time for all calibrators
    # Calibrator windows, as fraction of a day
    ts = np.array([0.0000, 0.1639, 0.3805, 0.6847, 0.7625, 0.8160])+t0
    te = np.array([0.1410, 0.3049, 0.6847, 0.7278, 0.8160, 1.1188])+t0
    # Day numbers of source transitions (this will shift for leap-years,
    # but it is not critical, so that is ignored.
    pcal2trans = np.array([[0,20,45,83,251,366],[0,75,83,251,291,309,330,366]])
    pcal3trans = np.array([[0,83,182,199,251,366],[0,83,111,128,197,251,366],[0,83,132,251,366]])
    refcaltrans = np.array([[0,121,210,242,305,366],[0,31,37,84,210,366]])
    # Nominal phasecal scan start times
    pcal2s = np.array([(17. + 10./60.)/24.,(22. + 10./60.)/24.])  # 17:10 and 22:10 UT
#    pcal3s = np.array([(15. + 10./60.)/24.,(19. + 50./60.)/24.,(24. + 10./60.)/24.]) # 15:10, 19:50 and 24:10 UT
# For summer 2023, shift the last calibrator earlier, to complete before 24:00 UT and maximize overlap with SRH
    pcal3s = np.array([(15. + 10./60.)/24.,(19. + 50./60.)/24.,(23. + 10./60.)/24.]) # 15:10, 19:50 and 23:10 UT
    # Source designations for each range
    pcal2srcs  = [[3,4,5,-1,2],[5,0,-1,2,3,4,5]] 
    pcal3srcs  = [[-1,5,0,1,-1],[-1,5,0,1,2,-1],[-1,1,2,-1]]
    refcalsrcs = [[2,5,0,1,2],[5,0,1,2,5]]
    calnames = ['0319+415','0319+415', '1229+020', '1331+305', '2136+006', '2253+161']
#    for imjd = range(mjd0,mjd1+1):
    iday = int(imjd - mjd0)
    nday = imjd - int(sun['date'][0].mjd)   # Number of days into sun dictionary corresponding to this day
    # For each day, identify the sources
    #   Morning refcal
    refcal1 = refcalsrcs[0][np.where(iday - np.array(refcaltrans[0]) >= 0.0)[0][-1]]
    refset = (te[refcal1]-nday*0.0027378 + 1.0) % 1
    sunrise = sun['taz_rise'][nday].mjd % 1
    if iday > 130 and iday < 212:
        sunrise += 600./86400.
    rc1end = min(refset,sunrise)
    rc1start = rc1end - refdur/1440.
    lines = []
    lines.append('{:} {:} {:}'.format(Time(imjd + rc1start,format='mjd').iso[:19],'ACQUIRE',calnames[refcal1]))
    lines.append('{:} {:}'.format(Time(imjd + rc1start + 1./1440.,format='mjd').iso[:19],'LOSELECT'))
    lines.append('{:} {:} {:} {:}'.format(Time(imjd + rc1start + 4./1440.,format='mjd').iso[:19],'PHASECAL_LO',calnames[refcal1],'pcal_lo.fsq'))
    lines.append('{:} {:}'.format(Time(imjd + rc1start + 24./1440.,format='mjd').iso[:19],'HISELECT'))
    lines.append('{:} {:} {:} {:}'.format(Time(imjd + rc1start + 25./1440.,format='mjd').iso[:19],'PHASECAL',calnames[refcal1],'pcal_hi-all.fsq'))
    if verbose:
        print Time(imjd + rc1start,format='mjd').iso[:19],'ACQUIRE',calnames[refcal1]
        print Time(imjd + rc1start + 1./1440.,format='mjd').iso[:19],'LOSELECT'
        print Time(imjd + rc1start + 4./1440.,format='mjd').iso[:19],'PHASECAL_LO',calnames[refcal1],'pcal_lo.fsq'
        print Time(imjd + rc1start + 24./1440.,format='mjd').iso[:19],'HISELECT'
        print Time(imjd + rc1start + 25./1440.,format='mjd').iso[:19],'PHASECAL',calnames[refcal1],'pcal_hi-all.fsq'
    if rc1end != sunrise:
        lines.append('{:} {:}'.format(Time(imjd + rc1start + 85./1440.,format='mjd').iso[:19],'STOW'))
        if verbose: print Time(imjd + rc1start + 85./1440.,format='mjd').iso[:19],'STOW'
    if ant3_late:
        sun1cmd = 'SUN_NO_ANT3'
    else:
        sun1cmd = 'SUN'
    lines.append('{:} {:}'.format(Time(imjd + sunrise,format='mjd').iso[:19],sun1cmd))
    isun1 = len(lines) - 1
    if verbose: print Time(imjd + sunrise,format='mjd').iso[:19],sun1cmd
    ant3rise = None
    ant3set = None
    if ant3_late:
        t3rise, t3set = sun_limit_times(Time(imjd,format='mjd'), ant3_sun_limit)
        if t3rise is not None:
            # Never earlier than 1 min after the (Black-Mountain-adjusted) sunrise line
            ant3rise = max(t3rise.mjd % 1, sunrise + 1./1440.)
        if t3set is not None:
            ant3set = t3set.mjd % 1
            if ant3set < 0.5: ant3set += 1.0
    if ax:
        ax.plot([rc1start,rc1start+refdur/1440.],[imjd,imjd],color='C0',alpha=0.25)
    #   Phasecals
    if iday < 83 or iday >= 251:
        #import pdb; pdb.set_trace()
        pcal1 = pcal2srcs[0][np.where(iday - np.array(pcal2trans[0]) >= 0.0)[0][-1]]
        pc1rise = (ts[pcal1]-nday*0.0027378 + 1.0) % 1
        pc1set = (te[pcal1]-nday*0.0027378 + 1.0) % 1
        if pc1set < 0.5: pc1set += 1.0
        #print Time(pc1rise+imjd,format='mjd').iso, pc1set
        pc1start = max(pc1rise,pcal2s[0])
        #print Time(pc1start+imjd,format='mjd').iso
        pc1start = min(pc1start,pc1set-caldur/1440.)
        #print Time(pc1start+imjd,format='mjd').iso,Time(pc1set+imjd,format='mjd').iso
        #print pc1rise, pc1set, Time(imjd + pc1rise,format='mjd').iso,Time(imjd + pc1set,format='mjd').iso,
        pcal2 = pcal2srcs[1][np.where(iday - np.array(pcal2trans[1]) >= 0.0)[0][-1]]
        pc2rise = (ts[pcal2]-nday*0.0027378 + 1.0) % 1
        pc2set = (te[pcal2]-nday*0.0027378 + 1.0) % 1
        if pc2set < 0.5: pc2set += 1.0
        pc2start = max(pc2rise,pcal2s[1])
        pc2start = min(pc2start,pc2set-caldur/1440.)
        lines.append('{:} {:} {:}'.format(Time(imjd + pc1start,format='mjd').iso[:19],'ACQUIRE',calnames[pcal1]))
        lines.append('{:} {:} {:} {:}'.format(Time(imjd + pc1start + 4./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal1],'pcal_hi-all.fsq'))
#        lines.append('{:} {:} {:} {:}'.format(Time(imjd + pc1start + 20./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal1],'solar.fsq'))
#        lines.append('{:} {:} {:}'.format(Time(imjd + pc1start + 21./1440.,format='mjd').iso[:19],'SKYCALTEST',calnames[pcal1]))
        lines.append('{:} {:}'.format(Time(imjd + pc1start + 20./1440.,format='mjd').iso[:19],'SUN'))
        lines.append('{:} {:}'.format(Time(imjd + (18.*60. + 30.)/1440.,format='mjd').iso[:19],'SOLPNTCAL solar.fsq'))
        lines.append('{:} {:}'.format(Time(imjd + (18.*60. + 37.)/1440.,format='mjd').iso[:19],'GAINSOLPNT'))
        lines.append('{:} {:}'.format(Time(imjd + (18.*60. + 39.)/1440.,format='mjd').iso[:19],'SUN'))
        #lines.append('{:} {:}'.format(Time(imjd + (20.*60. + 00.)/1440.,format='mjd').iso[:19],'GAINCALTEST'))
        #lines.append('{:} {:}'.format(Time(imjd + (20.*60. + 03.)/1440.,format='mjd').iso[:19],'SUN'))
        lines.append('{:} {:}'.format(Time(imjd + (21.*60. + 30.)/1440.,format='mjd').iso[:19],'SOLPNTCAL solar.fsq'))
        lines.append('{:} {:}'.format(Time(imjd + (21.*60. + 37.)/1440.,format='mjd').iso[:19],'GAINSOLPNT'))
        lines.append('{:} {:}'.format(Time(imjd + (21.*60. + 39.)/1440.,format='mjd').iso[:19],'SUN'))
        lines.append('{:} {:} {:}'.format(Time(imjd + pc2start,format='mjd').iso[:19],'ACQUIRE',calnames[pcal2]))
        lines.append('{:} {:} {:} {:}'.format(Time(imjd + pc2start + 4./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal2],'pcal_hi-all.fsq'))
        lines.append('{:} {:}'.format(Time(imjd + pc2start + 20./1440.,format='mjd').iso[:19],'SUN'))
#        lines.append('{:} {:}'.format(Time(imjd + pc2start + 35./1440.,format='mjd').iso[:19],'SUN'))
        if verbose:
            print Time(imjd + pc1start,format='mjd').iso[:19],'ACQUIRE',calnames[pcal1]
            print Time(imjd + pc1start + 4./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal1],'pcal_hi-all.fsq'
#            print Time(imjd + pc1start + 20./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal1],'solar.fsq'
#            print Time(imjd + pc1start + 21./1440.,format='mjd').iso[:19],'SKYCALTEST',calnames[pcal1]
            print Time(imjd + pc1start + 20./1440.,format='mjd').iso[:19],'SUN'
            print Time(imjd + (18.*60. + 30.)/1440.,format='mjd').iso[:19],'SOLPNTCAL solar.fsq'
            print Time(imjd + (18.*60. + 37.)/1440.,format='mjd').iso[:19],'GAINSOLPNT'
            print Time(imjd + (18.*60. + 39.)/1440.,format='mjd').iso[:19],'SUN'
            #print Time(imjd + (20.*60. + 00.)/1440.,format='mjd').iso[:19],'GAINCALTEST'          
            #print Time(imjd + (20.*60. + 03.)/1440.,format='mjd').iso[:19],'SUN'
            print Time(imjd + (21.*60. + 30.)/1440.,format='mjd').iso[:19],'SOLPNTCAL solar.fsq'
            print Time(imjd + (21.*60. + 37.)/1440.,format='mjd').iso[:19],'GAINSOLPNT'
            print Time(imjd + (21.*60. + 39.)/1440.,format='mjd').iso[:19],'SUN'
            print Time(imjd + pc2start,format='mjd').iso[:19],'ACQUIRE',calnames[pcal2]
            print Time(imjd + pc2start + 4./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal2],'pcal_hi-all.fsq'
            print Time(imjd + pc2start + 20./1440.,format='mjd').iso[:19],'SUN'
#            print Time(imjd + pc2start + 35./1440.,format='mjd').iso[:19],'SUN'
        if ax:
            ax.plot([pc1start,pc1start+caldur/1440.],[imjd,imjd],color='C0',alpha=0.25)
            ax.plot([pc2start,pc2start+caldur/1440.],[imjd,imjd],color='C0',alpha=0.25)
    else:
        pcal1 = pcal3srcs[0][np.where(iday - np.array(pcal3trans[0]) >= 0.0)[0][-1]]
        pc1rise = (ts[pcal1]-nday*0.0027378 + 1.0) % 1
        pc1set = (te[pcal1]-nday*0.0027378 + 1.0) % 1
        pc1start = max(pc1rise,pcal3s[0])
        pc1start = min(pc1start,pc1set-caldur/1440.)
        pcal2 = pcal3srcs[1][np.where(iday - np.array(pcal3trans[1]) >= 0.0)[0][-1]]
        pc2rise = (ts[pcal2]-nday*0.0027378 + 1.0) % 1
        pc2set = (te[pcal2]-nday*0.0027378 + 1.0) % 1
        if pc2set < 0.5: pc2set += 1.0
        pc2start = max(pc2rise,pcal3s[1])
        pc2start = min(pc2start,pc2set-caldur/1440.)
        pcal3 = pcal3srcs[2][np.where(iday - np.array(pcal3trans[2]) >= 0.0)[0][-1]]
        pc3rise = (ts[pcal3]-nday*0.0027378 + 1.0) % 1
        if pc3rise < 0.5: pc3rise += 1.0
        pc3set = (te[pcal3]-nday*0.0027378 + 1.0) % 1
        if pc3set < 0.5: pc3set += 1.0
        pc3start = max(pc3rise,pcal3s[2])
        pc3start = min(pc3start,pc3set-caldur/1440.)

        lines.append('{:} {:} {:}'.format(Time(imjd + pc1start,format='mjd').iso[:19],'ACQUIRE',calnames[pcal1]))
        lines.append('{:} {:} {:} {:}'.format(Time(imjd + pc1start + 4./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal1],'pcal_hi-all.fsq'))
#        lines.append('{:} {:} {:} {:}'.format(Time(imjd + pc1start + 20./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal1],'solar.fsq'))
#        lines.append('{:} {:} {:}'.format(Time(imjd + pc1start + 21./1440.,format='mjd').iso[:19],'SKYCALTEST',calnames[pcal1]))
        lines.append('{:} {:}'.format(Time(imjd + pc1start + 20./1440.,format='mjd').iso[:19],'SUN'))
        #lines.append('{:} {:}'.format(Time(imjd + (17.*60. + 00.)/1440.,format='mjd').iso[:19],'GAINCALTEST'))
        #lines.append('{:} {:}'.format(Time(imjd + (17.*60. + 03.)/1440.,format='mjd').iso[:19],'SUN'))
        lines.append('{:} {:}'.format(Time(imjd + (18.*60. + 30.)/1440.,format='mjd').iso[:19],'SOLPNTCAL solar.fsq'))
        lines.append('{:} {:}'.format(Time(imjd + (18.*60. + 37.)/1440.,format='mjd').iso[:19],'GAINSOLPNT'))
        lines.append('{:} {:}'.format(Time(imjd + (18.*60. + 39.)/1440.,format='mjd').iso[:19],'SUN'))
        lines.append('{:} {:} {:}'.format(Time(imjd + pc2start,format='mjd').iso[:19],'ACQUIRE',calnames[pcal2]))
        lines.append('{:} {:} {:} {:}'.format(Time(imjd + pc2start + 4./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal2],'pcal_hi-all.fsq'))
        lines.append('{:} {:}'.format(Time(imjd + pc2start + 20./1440.,format='mjd').iso[:19],'SUN'))
#        lines.append('{:} {:}'.format(Time(imjd + pc2start + 35./1440.,format='mjd').iso[:19],'SUN'))
        lines.append('{:} {:}'.format(Time(imjd + (21.*60. + 30.)/1440.,format='mjd').iso[:19],'SOLPNTCAL solar.fsq'))
        lines.append('{:} {:}'.format(Time(imjd + (21.*60. + 37.)/1440.,format='mjd').iso[:19],'GAINSOLPNT'))
        lines.append('{:} {:}'.format(Time(imjd + (21.*60. + 39.)/1440.,format='mjd').iso[:19],'SUN'))
        lines.append('{:} {:} {:}'.format(Time(imjd + pc3start,format='mjd').iso[:19],'ACQUIRE',calnames[pcal3]))
        lines.append('{:} {:} {:} {:}'.format(Time(imjd + pc3start + 4./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal3],'pcal_hi-all.fsq'))
        lines.append('{:} {:}'.format(Time(imjd + pc3start + 20./1440.,format='mjd').iso[:19],'SUN'))
#        lines.append('{:} {:}'.format(Time(imjd + pc3start + 35./1440.,format='mjd').iso[:19],'SUN'))
        if verbose:
            print Time(imjd + pc1start,format='mjd').iso[:19],'ACQUIRE',calnames[pcal1]
            print Time(imjd + pc1start + 4./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal1],'pcal_hi-all.fsq'
#            print Time(imjd + pc1start + 20./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal1],'solar.fsq'
#            print Time(imjd + pc1start + 21./1440.,format='mjd').iso[:19],'SKYCALTEST',calnames[pcal1]
            print Time(imjd + pc1start + 20./1440.,format='mjd').iso[:19],'SUN'
            #print Time(imjd + (17.*60. + 00.)/1440.,format='mjd').iso[:19],'GAINCALTEST'          
            #print Time(imjd + (17.*60. + 03.)/1440.,format='mjd').iso[:19],'SUN'
            print Time(imjd + (18.*60. + 30.)/1440.,format='mjd').iso[:19],'SOLPNTCAL solar.fsq'
            print Time(imjd + (18.*60. + 37.)/1440.,format='mjd').iso[:19],'GAINSOLPNT'
            print Time(imjd + (18.*60. + 39.)/1440.,format='mjd').iso[:19],'SUN'
            print Time(imjd + pc2start,format='mjd').iso[:19],'ACQUIRE',calnames[pcal2]
            print Time(imjd + pc2start + 4./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal2],'pcal_hi-all.fsq'
            print Time(imjd + pc2start + 20./1440.,format='mjd').iso[:19],'SUN'
#            print Time(imjd + pc2start + 35./1440.,format='mjd').iso[:19],'SUN'
            print Time(imjd + (21.*60. + 30.)/1440.,format='mjd').iso[:19],'SOLPNTCAL solar.fsq'
            print Time(imjd + (21.*60. + 37.)/1440.,format='mjd').iso[:19],'GAINSOLPNT'
            print Time(imjd + (21.*60. + 39.)/1440.,format='mjd').iso[:19],'SUN'
            print Time(imjd + pc3start,format='mjd').iso[:19],'ACQUIRE',calnames[pcal3]
            print Time(imjd + pc3start + 4./1440.,format='mjd').iso[:19],'PHASECAL',calnames[pcal3],'pcal_hi-all.fsq'
            print Time(imjd + pc3start + 20./1440.,format='mjd').iso[:19],'SUN'
#            print Time(imjd + pc3start + 35./1440.,format='mjd').iso[:19],'SUN'
        if ax:
            ax.plot([pc1start,pc1start+caldur/1440.],[imjd,imjd],color='C0',alpha=0.25)
            ax.plot([pc2start,pc2start+caldur/1440.],[imjd,imjd],color='C0',alpha=0.25)
            ax.plot([pc3start,pc3start+caldur/1440.],[imjd,imjd],color='C0',alpha=0.25)
    if ant3_late and ant3rise is not None:
        # Insert the Ant 3 join line just after the first SUN line, unless the
        # Sun reaches the Ant 3 limit only around/after the first phasecal
        # block, in which case Ant 3 simply joins at that block's normal SUN.
        if ant3rise < pc1start - 2./1440.:
            lines.insert(isun1 + 1, '{:} {:}'.format(Time(imjd + ant3rise,format='mjd').iso[:19],'SUN_ANT3'))
            if verbose: print Time(imjd + ant3rise,format='mjd').iso[:19],'SUN_ANT3'
    #   Evening refcal
    refcal2 = refcalsrcs[1][np.where(iday - np.array(refcaltrans[1]) >= 0.0)[0][-1]]
    refrise = (ts[refcal2]-nday*0.0027378 + 1.0) % 1
    if refrise < 0.5: refrise += 1.0
    sunset = sun['taz_set'][nday].mjd % 1
    if sunset < 0.5: sunset += 1.0
    if ant3_late and ant3set is not None:
        # Park Ant 3 before its elevation demand falls below the soft limit.
        ant3set = min(ant3set, sunset - 1./1440.)
        lines.append('{:} {:}'.format(Time(imjd + ant3set,format='mjd').iso[:19],'STOW_ANT3'))
        if verbose: print Time(imjd + ant3set,format='mjd').iso[:19],'STOW_ANT3'
    rc2start = max(refrise,sunset)
#        rc2end = rc2start + refdur/1440.
    if refrise > sunset:
        lines.append('{:} {:}'.format(Time(imjd + sunset,format='mjd').iso[:19],'STOW'))
        if verbose: print Time(imjd + sunset,format='mjd').iso[:19],'STOW'
    lines.append('{:} {:} {:}'.format(Time(imjd + rc2start,format='mjd').iso[:19],'ACQUIRE',calnames[refcal2]))
    lines.append('{:} {:}'.format(Time(imjd + rc2start + 1./1440.,format='mjd').iso[:19],'LOSELECT'))
    lines.append('{:} {:} {:} {:}'.format(Time(imjd + rc2start + 4./1440.,format='mjd').iso[:19],'PHASECAL_LO',calnames[refcal2],'pcal_lo.fsq'))
    lines.append('{:} {:}'.format(Time(imjd + rc2start + 24./1440.,format='mjd').iso[:19],'HISELECT'))
    lines.append('{:} {:} {:} {:}'.format(Time(imjd + rc2start + 25./1440.,format='mjd').iso[:19],'PHASECAL',calnames[refcal2],'pcal_hi-all.fsq'))
    rewind_offset = 86. if ant13_fem_power else 85.
    lines.append('{:} {:}'.format(Time(imjd + rc2start + rewind_offset/1440.,format='mjd').iso[:19],'REWIND'))
    if verbose:
        print Time(imjd + rc2start,format='mjd').iso[:19],'ACQUIRE',calnames[refcal2]
        print Time(imjd + rc2start + 1./1440.,format='mjd').iso[:19],'LOSELECT'
        print Time(imjd + rc2start + 4./1440.,format='mjd').iso[:19],'PHASECAL_LO',calnames[refcal2],'pcal_lo.fsq'
        print Time(imjd + rc2start + 24./1440.,format='mjd').iso[:19],'HISELECT'
        print Time(imjd + rc2start + 25./1440.,format='mjd').iso[:19],'PHASECAL',calnames[refcal2],'pcal_hi-all.fsq'
        print Time(imjd + rc2start + rewind_offset/1440.,format='mjd').iso[:19],'REWIND'
    if ax:
        ax.plot([rc2start,rc2start+refdur/1440.],[imjd,imjd],color='C0',alpha=0.25)
    lines = chk_sched(lines)
    if ant13_fem_power:
        return add_ant13_fem_power_events(lines)
    return lines

def chk_sched(lines):
    # Post-creation check on the schedule to address the fact that
    # 3C273 is too close to the Sun from 9/15 to 10/13 each year (day-of-year 259-287).
    doy = int(Time(lines[0][:10]).yday[5:8])
    next = -1  # Impossible line number
    if doy >= 259 and doy <= 287:
        # This is a date range when the 27-m antenna should not point at source 1229+020
        # Simply replace 1229+020 with 1331+305 (3C286).  That is a much weaker source, but it
        # may be okay for a PHASECAL
        if doy < 272:
            dt = (3000. - (doy - 259)*240.)/86400.  # Starts at 50 min and decreases by 4 min each day
        else:
            dt = 0.
        for i in range(len(lines)):
            if lines[i].find('1229+020') != -1:
                # This line has to be adjusted by dt later and change source to '1331+305'
                lines[i] = lines[i].replace('1229+020','1331+305')
                lines[i] = (Time(lines[i][:19])+dt).iso[:19]+lines[i][19:]
                next = i+1  # Set to next line number, which also has to increment by dt
            elif next == i:
                lines[i] = (Time(lines[i][:19])+dt).iso[:19]+lines[i][19:]
                next = -1
    return lines

def remove_cal(lines, ant13_fem_power=False):
    '''Remove 27-m calibration blocks from a generated solar schedule.

    :param lines: Input schedule lines.
    :type lines: list(str)
    :param ant13_fem_power: Rebuild Ant 13 FEM power events when ``True``.
    :type ant13_fem_power: bool
    :returns: Filtered schedule lines.
    :rtype: list(str)
    '''
    from util import Time
    keepidx = []
    sunidx = []
    rmidx = []
    clean_lines = []
    last_acquire = None
    last_acquire_idx = None
    last_stow = None
    last_stow_idx = None
    for line in lines:
        tokens = line[20:].split()
        command = tokens[0].upper() if tokens else ''
        if command in ('FEMPOWERON', 'FEMPOWEROFF'):
            continue
        clean_lines.append(line)
        if command == 'ACQUIRE':
            last_acquire = line[:19]
            last_acquire_idx = len(clean_lines) - 1
    lines = clean_lines
    # Only the STOW directly before the final ACQUIRE ends the last solar
    # scan.  An earlier STOW can belong to the morning reference calibration.
    if last_acquire_idx is not None and last_acquire_idx > 0:
        tokens = lines[last_acquire_idx - 1][20:].split()
        command = tokens[0].upper() if tokens else ''
        if command == 'STOW':
            last_stow_idx = last_acquire_idx - 1
            last_stow = lines[last_stow_idx][:19]
    # Keep all SUN lines and the line following. Also keep GAINSOLPNT lines;
    # these should remain even in "No 27m" mode.  A STOW_ANT3 line (Ant 3
    # early stow) is kept along with the line following it, since it displaces
    # the array-wide STOW from the line-after-SUN position.
    for i,line in enumerate(lines):
        if line.find('SUN') > 0:
            keepidx.append(i)
            keepidx.append(i+1)
        if line.find('STOW_ANT3') > 0:
            keepidx.append(i)
            keepidx.append(i+1)
        if line.find('GAINSOLPNT') > 0:
            keepidx.append(i)
        if line.find('SKYCAL') > 0:
            keepidx.append(i-1)
            keepidx.append(i)
        if line.find('REWIND') > 0:
            keepidx.append(i)
    if ant13_fem_power and last_stow_idx is not None:
        if last_stow_idx not in keepidx:
            keepidx.append(last_stow_idx)
            keepidx.sort()
    # Deduplicate while preserving chronological order (e.g. a SUN_ANT3 line
    # matches the SUN rule and is also the line following SUN_NO_ANT3).
    keepidx = sorted(set(keepidx))
    keeplines = np.array(lines)
    keeplines = keeplines[keepidx]
    
    for i,line in enumerate(keeplines):
        if line.find('SUN') > 0:
            sunidx.append(i)
        if line.find('PHASECAL') > 0:
            # Subtract 1 min from time in this line and use that time in previous (ACQUIRE) line
            mjd = Time(line[:19]).mjd - 60./86400
            keeplines[i-1] = Time(mjd,format='mjd').iso[:19] + keeplines[i-1][19:]
    for i in sunidx[:-1]:   # Was for i in sunidx[1:-1]:, in order to skip first ACQUIRE
        if keeplines[i+1].find('ACQUIRE') > 1:
            rmidx.append(i+1)
            rmidx.append(i+2)
    outlines = []
    for i,line in enumerate(keeplines):
        tokens = line[20:].split()
        if len(tokens) > 0:
            cmd = tokens[0].upper()
        else:
            cmd = ''
        if i in rmidx:
            continue
        if len(tokens) > 0:
            if cmd == 'ACQUIRE' or cmd.startswith('PHASECAL'):
                # "No 27m" mode: drop all refcal/phasecal blocks.
                continue
        outlines.append(line)
    if ant13_fem_power and last_stow:
        # With the 27-m calibration blocks removed, the last solar STOW is
        # the true end of observing.  Keep the terminal REWIND one minute
        # later so the FEMPOWEROFF macro can finish before it is due.
        rewind_mjd = Time(last_stow).mjd + 1./1440.
        outlines[-1] = Time(rewind_mjd,format='mjd').iso[:19]+' REWIND'
    elif last_acquire:
        rewind_delay = 1./1440. if ant13_fem_power else 0.
        rewind_mjd = Time(last_acquire).mjd + rewind_delay
        outlines[-1] = Time(rewind_mjd,format='mjd').iso[:19]+' REWIND'
    else:
        outlines[-1] = outlines[-1][:19]+' REWIND'
    if ant13_fem_power:
        return add_ant13_fem_power_events(outlines)
    return outlines
    
