$SCAN-STOP
$WAIT 2
$SUBARRAY default.antlist phasecal
$MK_TABLES pcal_tab #1
TRACKTABLE pcal_tab.radec
TRACK
$WAIT 2
$SCAN-START NODATA
