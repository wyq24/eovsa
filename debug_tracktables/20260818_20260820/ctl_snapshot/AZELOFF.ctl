$SCAN-STOP
FSEQ-OFF
$MK_TABLES sun_tab Sun
TRACKTABLE sun_tab.radec
TRACK ant1-15
$wait 2
FSEQ-FILE solar.fsq
FSEQ-ON
femauto-off ant1-15
femattn 0 0 ant1-15
$SCAN-START
$wait 10
azeloff -5 0
$wait 20
azeloff -2 0 
$wait 10
azeloff -1 0
$wait 10
azeloff -0.5 0.0
$wait 10
azeloff -0.2 0.0
$wait 10
azeloff -0.1 0.0
$wait 10
azeloff 0 0
$wait 10
azeloff 0.1 0.0
$wait 10
azeloff 0.2 0.0
$wait 10
azeloff 0.5 0.0
$wait 10
azeloff 1 0
$wait 10
azeloff 2 0
$wait 10
azeloff 5 0
$wait 10
azeloff 0 -5
$wait 20
azeloff 0 -2 
$wait 10
azeloff 0 -1
$wait 10
azeloff 0.0 -0.5
$wait 10
azeloff 0.0 -0.2
$wait 10
azeloff 0.0 -0.1
$wait 10
azeloff 0 0
$wait 10
azeloff 0.0 0.1
$wait 10
azeloff 0.0 0.2
$wait 10
azeloff 0.0 0.5
$wait 10
azeloff 0 1
$wait 10
azeloff 0 2
$wait 10
azeloff 0 5
$wait 10
azeloff 0 0
$wait 10
$SCAN-STOP
$wait 2
