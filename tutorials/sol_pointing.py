## summary how to use solpnt_x to do antenna pointing. 
## https://ovsa.njit.edu/wiki/index.php/Owen%27s_Notes#Update_Antenna_Pointing
import solpnt_x as sx
from util import Time
tsolpnt = Time('2025-10-24 21:30')
solout = sx.solpnt_xanal(tsolpnt)

sx.solpnt_bsize(solout)

offsets = sx.solpnt_offsets(solout)

## For example, the following command will adjust the pointing offsets for antennas 6 to 8, 10 and 13:
sx.offsets2ants(offsets, ant_str='ant6 ant7 ant8 ant10 ant13')
## The antennas that have had tracking updated will need to be rebooted. From the Schedule Window issue the commands:
# reboot 1 ant6-8

# tracktable sun_tab.radec 1 ant6-8 track ant6-8

# The antenna pointing adjustment is typically performed once per month.

