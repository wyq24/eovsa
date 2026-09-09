# History
# 2019-04-10  DG
#   Several changes to dpp_fix_packets() to that it never exits (CTRL-C to kill it),
#   and to fix a bug so that the timeout is honored correctly.  Also, the procstat()
#   routine is no longer called during a timeout.  Since the interfaces only need 
#   resetting every 15 minutes, the timeout is increased to 10 minutes.
# 2021-08-08  DG
#   Owen wrote alternative version, fix_packets2(), which rewrites the SMP_AFFINITY.sh
#   script on the fly when packets go to zero.  I added a couple of lines to remove
#   the empty /tmp/netplan_* folders created during a network reset.
# 2021-08-10  DG
#   Removing the files with a wild-card was not working, so now filenames are found
#   with glob() before removing by explicit name.
# 2021-08-25  OG
#   Modified update_log so that it writes to the /common/webplots/dpp_fix_packets_log.txt
#   file. This is so it can be accessed by the sf_display program. It only saves the
#   last 5 lines. This can be changed if necessary.
# 2026-09-01  SY
#   Measure the two data NICs directly and reset only the affected data link after
#   sustained packet loss.  Never run host-wide netplan apply, which also disrupts
#   the DHCP management interface. Keep a stable CPU affinity instead of rotating
#   it when both links are quiet, since that is also the normal idle state.

# 2026-09-06  SY
#   Report sustained low rates without cycling links; retain startup CPU setup.
#   Automatic link recovery introduced hard packet gaps during observations.
#   Keep both data MACs learned with small periodic probes. A one-port probe
#   restored the opposite RX stream, consistent with switch unicast flooding.

from __future__ import print_function

import datetime
import os
import time
import subprocess


DATA_INTERFACES = ('enp5s0', 'enp7s0')
COUNTERS = ('rx_packets', 'rx_dropped', 'rx_errors')
ACTIVE_PACKET_RATE = 100000.0
MIN_PACKET_RATE = 130000.0
BAD_SAMPLE_LIMIT = 5
LOG_COOLDOWN = 600
MAC_REFRESH_INTERVAL = 60
SYSFS_ROOT = '/sys/class/net'


def _utc_now_iso():
    return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


def read_interface_counters(interface, sysfs_root=SYSFS_ROOT):
    """Read packet counters for one network interface from sysfs.

    :param interface: Kernel interface name.
    :type interface: str
    :param sysfs_root: Root of the network-interface sysfs tree.
    :type sysfs_root: str
    :returns: RX packet, dropped-packet, and error counters.
    :rtype: dict
    :raises IOError: If an interface counter cannot be read.
    :raises ValueError: If a counter does not contain an integer.
    """
    counters = {}
    for name in COUNTERS:
        path = os.path.join(sysfs_root, interface, 'statistics', name)
        with open(path, 'r') as counter_file:
            counters[name] = int(counter_file.read().strip())
    return counters


def packet_rate(previous, current, elapsed):
    """Calculate an RX packet rate from two counter samples.

    :param previous: Earlier interface counters.
    :type previous: dict or None
    :param current: Current interface counters.
    :type current: dict or None
    :param elapsed: Seconds between samples.
    :type elapsed: float
    :returns: Packets per second, or ``None`` for an invalid/reset counter.
    :rtype: float or None
    """
    if previous is None or current is None or elapsed <= 0:
        return None
    delta = current['rx_packets'] - previous['rx_packets']
    if delta < 0:
        return None
    return float(delta) / elapsed


class SustainedPacketLoss(object):
    """Track consecutive low-rate samples independently for each data NIC."""

    def __init__(self, interfaces=DATA_INTERFACES,
                 active_rate=ACTIVE_PACKET_RATE,
                 minimum_rate=MIN_PACKET_RATE,
                 sample_limit=BAD_SAMPLE_LIMIT):
        self.interfaces = tuple(interfaces)
        self.active_rate = float(active_rate)
        self.minimum_rate = float(minimum_rate)
        self.sample_limit = int(sample_limit)
        self.bad_samples = dict((interface, 0)
                                for interface in self.interfaces)

    def observe(self, interface, rate, peer_rate=None):
        if interface not in self.bad_samples:
            raise ValueError('interface is not monitored: ' + interface)
        # Preserve the legacy trigger band exactly: 100,000 < rate < 130,000.
        if (rate is None or rate <= self.active_rate or
                rate >= self.minimum_rate):
            self.bad_samples[interface] = 0
            return False
        self.bad_samples[interface] += 1
        return self.bad_samples[interface] == self.sample_limit

    def reset(self, interface):
        self.bad_samples[interface] = 0


def init_fix_packets(cpu):
    #print "Reinitilizing fix_packets() using CPUs "+str(cpu[0]) +' and '+ str(cpu[1])
    
    #first need to edit the SMP_AFFINITY.sh
    smpaff='/home/user/test_svn/shell_scripts/SMP_AFFINITY.sh'
    f = open(smpaff,'r')
    lines = f.readlines()
    f.close()
    
    #comment out all the CPU lines
    for i,l in enumerate(lines):
        if '#(CPU' in l:
            if l[0] != '#':
                lines[i] = '#'+lines[i]
    
    #now uncomment the appropriate cpu lines
    for c in cpu:
        substring = '#(CPU '+str(c)+')'
        for i,l in enumerate(lines):
            if substring in l:
                if l[0] == '#':
                    lines[i] = lines[i][1:]
                    break
        
    f = open(smpaff,'w')
    for l in lines:
        f.write(l)
    f.close()
    
    #Now run the script
    command = ['bash','/home/user/test_svn/shell_scripts/SMP_AFFINITY.sh']
    proc = subprocess.Popen(command)

def update_log(msg):
    '''Reads each line of the file /common/webplots/dpp_fix_packets_log.txt into
    a list and appends the msg to the list. Only the last 20 lines are written
    back to the file'''
    
    logfile="/common/webplots/dpp_fix_packets_log.txt"
    f = open(logfile,'r')
    lines = f.readlines()
    f.close()
    lines.append(_utc_now_iso()+": "+msg+'\n')
    if len(lines) > 20:
        lines = lines[-20:]
    
    f = open(logfile,'w')
    for l in lines:
        f.write(l)
    f.close()

def update_cpu_file(cpu):
    '''writes out the cpu's currently being used to /common/webplots/dpp_cpu.txt'''
    
    cpufile="/common/webplots/dpp_cpu.txt"
    f = open(cpufile,'w')
    for c in cpu:
        f.write(str(c)+'\n')
        
    f.close()

def roach_hint(interface):
    '''Return a conservative hint about which ROACH group may need capture-based
    follow-up.

    The interface counter identifies the affected receive path, but not a single
    ROACH board. The board-level mapping available from packet capture tools is:

    - enp5s0 (formerly eth2) -> ROACH boards 1, 2, 5, 6
    - enp7s0 (formerly eth3) -> ROACH boards 3, 4, 7, 8

    This is only a hint for the operator. Exact board ID isolation still needs a
    packet capture.
    '''

    groups = {
        'enp5s0': 'ROACH boards 1,2,5,6',
        'enp7s0': 'ROACH boards 3,4,7,8',
    }
    return (interface + ' carries ' + groups.get(interface, 'an unknown group') +
            '; capture that interface to isolate the exact board')


def refresh_data_mac_learning():
    """Send small probes through both data ports to refresh MAC learning.

    These on-link ROACH data addresses do not normally answer ping. The ARP
    traffic advertises each DPP source MAC; an unanswered ping is expected.
    Both ports need refresh because flooding can affect the opposite RX port.
    This does not change interfaces, routes, or ROACH configuration.

    :returns: None.
    :rtype: None
    """
    targets = (('enp5s0', '10.0.1.11'), ('enp7s0', '10.0.2.31'))
    with open(os.devnull, 'w') as output:
        for interface, target in targets:
            status = subprocess.call(
                ['/bin/ping', '-n', '-I', interface, '-c', '1', '-W', '1', target],
                stdout=output, stderr=output)
            if status not in (0, 1):
                update_log('MAC-learning probe failed on ' + interface +
                           ': ping exit status ' + str(status))


def _sleep_to_next_second():
    time.sleep(max(0.0, 1.0 - (time.time() % 1.0)))


def monitor_packets():
    """Maintain MAC learning and report low RX rates without cycling links.

    :returns: Does not return during normal monitoring.
    :rtype: None
    """
    cpu = [18, 19]
    update_cpu_file(cpu)
    update_log('dpp_fix_packets.py has been restarted '
               '(MAC-learning refresh enabled; automatic interface resets disabled)')
    update_log('Using CPUs '+str(cpu[0])+' and '+str(cpu[1])+'.')
    init_fix_packets(cpu)
    refresh_data_mac_learning()

    detector = SustainedPacketLoss()
    previous = {}
    for interface in DATA_INTERFACES:
        previous[interface] = read_interface_counters(interface)
    previous_time = time.time()
    next_mac_refresh = previous_time + MAC_REFRESH_INTERVAL
    log_after = dict((interface, previous_time + 5.0)
                     for interface in DATA_INTERFACES)
    last_report_minute = None

    while True:
        _sleep_to_next_second()
        now = time.time()
        elapsed = now - previous_time
        rates = {}

        for interface in DATA_INTERFACES:
            try:
                current = read_interface_counters(interface)
                rate = packet_rate(previous.get(interface), current, elapsed)
            except (IOError, OSError, ValueError, KeyError) as error:
                current = None
                rate = None
                print(_utc_now_iso(), interface, 'counter read failed:', error)

            rates[interface] = rate
            if current is not None:
                previous[interface] = current

        for interface in DATA_INTERFACES:
            if now < log_after[interface]:
                detector.reset(interface)
                continue

            peer_interface = (DATA_INTERFACES[1]
                              if interface == DATA_INTERFACES[0]
                              else DATA_INTERFACES[0])
            rate = rates[interface]
            if detector.observe(interface, rate,
                                peer_rate=rates[peer_interface]):
                rounded_rate = int(round(rate))
                message = ('Sustained low packet rate on ' + interface + ': ' +
                           str(rounded_rate) + ' packets/s; '
                           'no interface reset; ' +
                           roach_hint(interface))
                print(_utc_now_iso(), message)
                update_log(message)
                log_after[interface] = now + LOG_COOLDOWN
                detector.reset(interface)

        previous_time = now
        report_minute = int(now // 60)
        if report_minute != last_report_minute:
            print(_utc_now_iso(), 'RX packet rates:',
                  ', '.join(interface + '=' +
                            ('unavailable' if rates[interface] is None else
                             str(int(round(rates[interface]))))
                            for interface in DATA_INTERFACES))
            last_report_minute = report_minute

        if now >= next_mac_refresh:
            refresh_data_mac_learning()
            next_mac_refresh = time.time() + MAC_REFRESH_INTERVAL


def main():
    monitor_packets()


if __name__ == '__main__':
    main()
