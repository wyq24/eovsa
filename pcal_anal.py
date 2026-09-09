#
# History:
#  2017-Mar-05 YC
#    Initially written by Yi Chai
#  2017-Mar-30 DG
#    Extensive rewrites to streamline code and fix some bugs
#  2017-Jul-11 DG
#    Changed time plot frequency to middle frequency of range, and label 
#    frequency on plot
#  2018-Jan-02 DG
#    Added mv_pcal_files().  Also fixed a bug in graph(), to avoid
#    a crash when a bad/short IDB file is analyzed. 
#  2018-Jun-09 DG
#    Change plot to plot phases of antennas other than ant1 relative to ant1
#  2019-May-04 DG
#    Added mv_ptg_files().
#  2019-Jun-21 DG
#    Fixed a bug in filenames when running on pipeline.
#  2020-Jan-26 DG
#    mv_pcal_files() only worked for years 201*, so now works for 20*
#  2020-May-29 SY
#    update calpntanal() to use util.get_idbdir() to find IDB root path.
#  2021-Aug-02 DG
#    Fix a bug in findfile() introduced when FDB files were lost from the DPP.
#    Now on pipeline the IFDB files are used, which do not have a key ST_SEC.
#  2022-Mar-08 DG
#    Temporarily commented out check for Windscram due to loss of SQL
#  2023-Oct-29 DG
#    Added back the Windscram check.
#  2024-Jun-17 DG
#    Belatedly added back the windscram check.
#  2025-May-21  DG
#    Changes to work with 16 antennas.  It is mainly just a larger plot with more
#    antennas.
#  2026-Sep-03  SY
#    Fix pcal_anal() so that a scan spanning several IDB files is re-processed
#    once all of its files are complete.  findfile() marks the last file of a
#    scan "undone" until 10 min after the scan ends, so the "active scan" pass
#    wrote the NPZ/PNG products from the files closed so far, and the later
#    "completed scan" pass then skipped the scan because the pcT*.png plot
#    already existed.  The last IDB file of every multi-file scan (half of a
#    20-min PHASECAL_LO scan) therefore never made it into the NPZ.  graph()
#    now writes a pcL<time>_<source>.txt marker listing the IDB files it used,
#    and the completed-scan pass re-runs graph() unless the marker already
#    covers every file of the scan.  Also fixed findfile() to index the
#    in-range scans directly; it assumed all out-of-range scans came before
#    the in-range ones, which returned the wrong scans for a timerange that
#    ends before a later scan on the same day (the cron, whose range ends
#    at "now", was not affected).
#

import numpy as np
from util import Time, lobe, fname2mjd,get_idbdir
ten_minutes = 600./86400.
one_minute = 60./86400.
PHASECAL_WEBROOT = '/nas8/eovsa/phasecal/'

def mv_pcal_files():
    ''' Moves (renames) files in the phasecal web folder
        into new folders according to date.  Leaves the last 20 .npz
        and associated files in the main folder.
    '''
    import glob, os
    from time import sleep
    npzfiles = glob.glob(os.path.join(PHASECAL_WEBROOT, '20????????*'))
    npzfiles.sort()
    #datstr = ''
    if len(npzfiles) > 20:
        for file in npzfiles[:-20]:
            #datstr_prev = datstr
            datstr = os.path.basename(file)[:8]
            year = datstr[:4]
            #if datstr != datstr_prev:
            directory = os.path.join(PHASECAL_WEBROOT, year, datstr)
            if not os.path.exists(directory):
                #print 'mkdir',directory 
                os.makedirs(directory)
                sleep(0.1)
            #print 'mv',file,directory+os.path.basename(file) 
            os.rename(file, os.path.join(directory, os.path.basename(file)))
            files = glob.glob(os.path.join(PHASECAL_WEBROOT, 'pc?'+datstr+'*'))
            files.sort()    
            for f in files:
                #print 'mv',f,directory+os.path.basename(f) 
                os.rename(f, os.path.join(directory, os.path.basename(f)))

def mv_ptg_files():
    ''' Moves (renames) files in the /common/webplots/PTG folder
        into new folders according to date.  Leaves the last 20 PTG
        files in the main folder.
    '''
    import glob, os
    from time import sleep
    files = glob.glob('/common/webplots/PTG/P*.png')
    files.sort()
    #datstr = ''
    if len(files) > 20:
        for file in files[:-20]:
            #datstr_prev = datstr
            datstr = file[26:30]
            #if datstr != datstr_prev:
            directory = file[:21]+file[24:30]+'/'
            if not os.path.exists(directory):
                #print 'mkdir',directory 
                os.makedirs(directory)
                sleep(0.1)
            #print 'mv',file,directory+os.path.basename(file) 
            os.rename(file,directory+os.path.basename(file))

def findfile(trange):

    from util import nearest_val_idx
    import struct, time, glob, sys, socket
    import dump_tsys

    host = socket.gethostname()
    if host == 'dpp':
        fpath = '/data1/IDB/'
    else:
        fpath = get_idbdir(trange[0])
    t1 = str(trange[0].mjd)
    t2 = str(trange[1].mjd)
    tnow = Time.now()

    if t1[:5] != t2[:5]:
        # End day is different than start day, so read and concatenate two fdb files
        fdb = {}
        fdb1 = dump_tsys.rd_fdb(trange[0])
        fdb2 = dump_tsys.rd_fdb(trange[1])
        for key in fdb1.keys():
            fdb.update({key:np.append(fdb1[key],fdb2[key])})
    else:
        # Both start and end times are on the same day
        fdb = dump_tsys.rd_fdb(trange[0])

    scanidx, = np.where(fdb['PROJECTID'] == 'PHASECAL')
    scans,sidx = np.unique(fdb['SCANID'][scanidx],return_index=True)
    eidx = np.append(sidx[1:],len(scanidx)) - 1
    # List of PHASECAL scan start times
    tslist = Time(fdb['ST_TS'][scanidx[sidx]].astype(float).astype(int),format='lv')
    # List of PHASECAL scan end times
    telist = Time(fdb['EN_TS'][scanidx[eidx]].astype(float).astype(int),format='lv')
    # Remove any bad values (i.e. those with ST_SEC = 0)
    try:
        good, = np.where(fdb['ST_SEC'][scanidx[sidx]] != '0')
        tslist = tslist[good]
        telist = telist[good]
        scans = scans[good]
        sidx = sidx[good]
        eidx = eidx[good]
    except KeyError:
        # Key 'ST_SEC' not found so just continue (this happens on pipeline when IFDB file is used)
        pass
        
    flist = []
    status = []
    tstlist = []
    # Indexes of the scans that fall entirely within the timerange.  (An
    # earlier version counted the out-of-range scans and assumed they all
    # preceded the in-range ones, which picked the wrong scans whenever the
    # end of the timerange was earlier than a later scan on the same day.)
    inrange = [i for i in range(len(tslist))
                 if tslist[i].jd >= trange[0].jd and telist[i].jd <= trange[1].jd]
    k = len(inrange)  # Number of scans within timerange

    if k == 0:
        print 'No phase calibration data within given time range'
        return None
    else:
        print 'Found',k,'scans in timerange.'
        for i in inrange:
            f1 = fdb['FILE'][np.where(fdb['SCANID'] == scans[i])].astype('str')
            # if fpath == '/data1/eovsa/fits/IDB/':
            #     f2 = [fpath + f[3:11] + '/' + f for f in f1]
            # else:
            #     f2 = [fpath + f for f in f1]
            if host == 'dpp':
                f2 = [fpath + f for f in f1]
            else:
                f2 = [fpath + f[3:11] + '/' + f for f in f1]
            flist.append(f2)
            tstlist.append(tslist[i])
            ted = telist[i]
            # Mark all files done except possibly the last
            fstatus = ['done']*len(f1)
            # Check if last file end time is less than 10 min ago
            if (tnow.jd - ted.jd) < (600./86400):
                # Current time is less than 10 min after this scan
                fstatus[-1] = 'undone'
            status.append(fstatus)

    return {'scanlist':flist,'status':status,'tstlist':tstlist}

def graph(f,navg=None,path=None):

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import struct, time, glob, sys, socket, os
    import read_idb as ri
    import dbutil as db

    if navg is None:
        navg = 60

    if path is None:
        path = ''

    out = ri.read_idb(f,navg=navg)
    if out is None or not isinstance(out, dict) or 'fghz' not in out or len(out['fghz']) == 0:
        # This file is no good, so skip it
        return
    if out['time'][0] < Time('2025-05-22').jd:
        nsolant = 13
    else:
        nsolant = 15
    fig, ax = plt.subplots(4,nsolant,sharex=True, sharey=True)
    trange = Time([fname2mjd(f[0]),fname2mjd(f[-1]) + ten_minutes],format='mjd')
    times, wscram, avgwind = db.a14_wscram(trange)
    nwind = len(wscram)
    nbad = np.sum(wscram)
    if nbad != 0:
        warn = ' --> Windscram! ('+str(nbad)+' of '+str(nwind)+')'
        color = '#d62728'   # Plot points with "warning" Red color
    else:
        warn = ''
        color = '#1f77b4'   # Plot points with "normal" Blue color
    fig.set_size_inches(nsolant+5,6)
    nf = len(out['fghz'])
    fstr = str(out['fghz'][nf/2]*1000)[:5]+' MHz '
    for k in range(nsolant):
        for j in range(4):
            ax[j,k].cla()
            ax[j,k].plot(out['ha'],np.angle(out['x'][ri.bl2ord[k,nsolant],j,nf/2]),'.',color=color)
            ax[j,k].set_ylim(-4, 4)
            ax[j,k].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if k in range(1,nsolant): ax[j,k].yaxis.set_visible(False)
            if j in range(3): ax[j,k].xaxis.set_visible(False)
            if j == 0: 
                if k==0:
                    ax[0,k].title.set_text('Ant%d' %(k+1))
                else:
                    ax[0,k].title.set_text('Ant%d-Ant1' %(k+1))
    fig.suptitle(out['source']+'  '+Time(out['time'][0],format='jd').iso[:19]+' UT  '+fstr+warn)
    ax[0,0].set_ylabel('XX Phase')
    ax[1,0].set_ylabel('YY Phase')
    ax[2,0].set_ylabel('XY Phase')
    ax[3,0].set_ylabel('YX Phase')
    fig.text(0.5, 0.04, 'Hour Angle', ha = 'center')
    t = Time(out['time'][0],format='jd').iso[:19].replace('-','').replace(':','').replace(' ','')
    s = out['source']
    ofile = path + t[:14] +'_'+ s +'.npz'
    np.savez(open(ofile,'wb'),out = out)
    # Record which IDB files went into this product, so that pcal_anal() can
    # tell a partial (active-scan) product from a complete one.
    mfile = path + 'pcL' + t[:14] +'_'+ s +'.txt'
    fh = open(mfile,'w')
    fh.write('\n'.join([os.path.basename(str(fn)) for fn in f]) + '\n')
    fh.close()
    plt.savefig(path + 'pcT'+t+'_'+ s +'.png',bbox_inches='tight')
    plt.close(fig)

    ph = np.angle(np.sum(out['x'],3))
    fig, ax = plt.subplots(4,nsolant)
    fig.set_size_inches(nsolant+3,6)
    for k in range(nsolant):
        for j in range(4):
            ax[j,k].cla()
            if k == 0:
                ax[j,k].plot(out['fghz'],ph[ri.bl2ord[k,nsolant],j],'.',color=color)
            else:
                ax[j,k].plot(out['fghz'],lobe(ph[ri.bl2ord[k,nsolant],j]-ph[ri.bl2ord[0,nsolant],j]),'.',color=color)                
            ax[j,k].set_ylim(-4, 4)
            ax[j,k].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if k in range(1,nsolant): ax[j,k].yaxis.set_visible(False)
            if j in range(3): ax[j,k].xaxis.set_visible(False)
            if j == 0: 
                if k==0:
                    ax[0,k].title.set_text('Ant%d' %(k+1))
                else:
                    ax[0,k].title.set_text('Ant%d-Ant1' %(k+1))
    fig.suptitle(out['source']+' '+Time(out['time'][0],format='jd').iso[:19]+' UT'+warn)
    ax[0,0].set_ylabel('XX Phase')
    ax[1,0].set_ylabel('YY Phase')
    ax[2,0].set_ylabel('XY Phase')
    ax[3,0].set_ylabel('YX Phase')
    fig.text(0.5, 0.04, 'Frequency[GHz]', ha = 'center')
    t = Time(out['time'][0],format='jd').iso[:19].replace('-','').replace(':','').replace(' ','')
    plt.savefig(path + 'pcF'+t+'_'+ s +'.png',bbox_inches='tight')
    plt.close(fig)

    
def find_markers(path,first_file):
    ''' Return the pcL*.txt marker files written by graph() for the scan whose
        first IDB file is first_file.  The marker name carries the time of the
        first averaged sample, which can fall in the minute after the file
        time, so look at the file's minute and one minute either side of it.
    '''
    import glob
    tmark = fname2mjd(first_file)
    markers = []
    for t in (tmark - one_minute, tmark, tmark + one_minute):
        tstr = Time(t,format='mjd').iso.replace('-','').replace(':','').replace(' ','')[:12]
        markers += glob.glob(path + 'pcL' + tstr + '*.txt')
    return markers

def pcal_anal(trange,path=None):

    import os
    import os.path
    import socket
    import glob

    if path is None:
        path = ''

    out = findfile(trange)
    if out is None:
        return
    host = socket.gethostname()
    filelist = out['scanlist']
    statuslist = out['status']
    starttimelist = out['tstlist']
    nscans = len(filelist)
    print 'Found',nscans,'scans to process.'
    for i in range(nscans):
        good, = np.where(np.array(statuslist[i]) == 'done')
        flist = np.array(filelist[i])[good].tolist()   # List of "done" files
        first_file = filelist[i][0]
        if len(good) == len(filelist[i]):
            # All files in this scan are marked "done" (findfile() only marks
            # the last file done 10 min after the scan ends).  Process the scan
            # unless an earlier pass already made products from ALL of its
            # files.  The "active scan" pass below writes products from the
            # files closed at the time, so the existence of the plots is not
            # proof that the scan is complete; instead compare the scan's file
            # list with the pcL*.txt marker(s) written by graph().
            wanted = set([os.path.basename(str(fn)) for fn in filelist[i]])
            have = set()
            for mfile in find_markers(path,first_file):
                try:
                    have |= set(open(mfile).read().split())
                except IOError:
                    pass
            missing = wanted - have
            if not missing:
                print 'Scan processing already complete.  Skipping scan',i+1
            else:
                print 'Processing completed scan',i+1,'(',len(missing),'of',len(wanted),'files not yet in the products)'
                graph(filelist[i],path=path)
        elif len(good) == len(filelist[i])-1:
            # This scan is still active, so process all files up to this point.
            print 'Processing active scan',i+1
            if flist != []:
                graph(flist,path=path)
            else:
                print 'No files to process (yet).'

            
if __name__ == '__main__':
    # This code is running on dpp as a cron job using the shell script 
    # /home/user/test_svn/shell_scripts/pcal_anal.sh
    import matplotlib
    matplotlib.use('Agg')
    path=PHASECAL_WEBROOT
    t1 = Time.now().jd-0.25
    t2 = Time.now().jd
    trange = Time([t1,t2],format='jd')
    print trange.iso
    pcal_anal(trange,path=path)
