if {![string match "2022.2*" [version -short]]} {error "Vivado 2022.2 required"}
if {$argc != 2} {error "Expected routed DCP and new report path"}
set out [file normalize [lindex $argv 1]]
if {[file exists $out]} {error "Report already exists"}
set_param general.maxThreads 1
open_checkpoint [file normalize [lindex $argv 0]]
report_bus_skew -file $out
close_design
exit
