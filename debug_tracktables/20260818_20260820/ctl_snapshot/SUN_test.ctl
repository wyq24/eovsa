$SCAN-STOP
$WAIT 2
FLUSH ant16
FSEQ-OFF
FSEQ-INIT
DCMAUTO-OFF
$SUBARRAY default.antlist sun
$MK_TABLES sun_tab Sun
TRACKTABLE sun_tab.radec ant1-8 ant12
$WAIT 15
$MK_TABLES sun_tab09 Sun
TRACKTABLE sun_tab09.radec ant9
$WAIT 15
$MK_TABLES sun_tab10 Sun
TRACKTABLE sun_tab10.radec ant10
$WAIT 15
$MK_TABLES sun_tab11 Sun
TRACKTABLE sun_tab11.radec ant11
$WAIT 15
$MK_TABLES sun_tab13 Sun
TRACKTABLE sun_tab13.radec ant13
TRACK ant1-15
FSEQ-FILE solar.fsq
FSEQ-ON
STOW ant16
DCMAUTO-ON
AGC 0
$WAIT 5
$SCAN-START
