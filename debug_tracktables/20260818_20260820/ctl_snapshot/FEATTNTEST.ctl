# Sends all antennas to an RFI-quiet part of the sky, ready for 
# FEM/DCM attenuation adjustments
$SCAN-STOP
$WAIT 2
DCMAUTO-OFF ANT1-16
FSEQ-OFF
FSEQ-INIT
$SUBARRAY default.antlist phasecal
$MK_TABLES geosat_tab #1
TRACKTABLE geosat_tab.radec
TRACK
FSEQ-FILE #2
FSEQ-ON
FEMAUTO-OFF
FEMATTN 0
DCMATTN 10 10 ANT1-16
# Set the Ant 16 FEM attenuation to fixed values for now
HATTN 0 12 ANT16
VATTN 0 12 ANT16
$WAIT 2
# Start scan but do not record data
$SCAN-START NODATA
