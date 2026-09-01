$SCAN-STOP
FSEQ-OFF
$MK_TABLES sun_tab Sun
TRACKTABLE sun_tab.radec
TRACK ant1-15
$wait 2
FSEQ-FILE solar.fsq
FSEQ-ON
$SCAN-START
$wait 10
femauto-off
$wait 2
femattn 15
$wait 10
femattn 0
$wait 10
femattn 1
$wait 10
femattn 2
$wait 10
femattn 3
$wait 10
femattn 4
$wait 10
femattn 5
$wait 10
femattn 6
$wait 10
femattn 7
$wait 10
femattn 8
$wait 10
femattn 0
$wait 10
$SCAN-STOP
$wait 2
