$SCAN-STOP
$WAIT 2
FLUSH ant16
FSEQ-OFF
FSEQ-INIT
DCMAUTO-OFF
$SUBARRAY default.antlist sun
$MK_TABLES sun_tab Sun
TRACKTABLE sun_tab.radec ant1-8 ant12
TRACKTABLE sun_tab_ant09.radec ant9
TRACKTABLE sun_tab_ant10.radec ant10
TRACKTABLE sun_tab_ant11.radec ant11
TRACKTABLE sun_tab_ant13.radec ant13
TRACK ant1-15
FSEQ-FILE solar.fsq
FSEQ-ON
STOW ant16
DCMAUTO-ON
AGC 0
$WAIT 5
$SCAN-START
