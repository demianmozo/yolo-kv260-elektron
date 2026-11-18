# Prepared for Vivado 2022.2. Not executed during preparation.
# Usage: vivado -mode batch -nolog -nojournal -source report_routed.tcl \
#          -tclargs /absolute/final_routed.dcp /absolute/new/reports
# Opens the existing checkpoint without re-running implementation or saving it.
if {$argc != 2} {
    error "Usage: report_routed.tcl FINAL_ROUTED_DCP NEW_REPORTS_DIRECTORY"
}
if {![string match "2022.2*" [version -short]]} {
    error "This reporting flow requires Vivado 2022.2"
}
set dcp [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
if {![file isfile $dcp] || [file extension $dcp] ne ".dcp"} {
    error "Checkpoint does not exist: $dcp"
}
if {[file exists $out]} {
    error "Use a new report directory to preserve previous evidence: $out"
}
set_param general.maxThreads 1
open_checkpoint $dcp
# Report routing explicitly: a successful open alone does not prove full routing.
file mkdir $out
set f [open [file join $out provenance.txt] w]
puts $f "Vivado: [version -short]"
puts $f "Source checkpoint: $dcp"
puts $f "Source size: [file size $dcp] bytes"
puts $f "Source SHA256: [exec sha256sum -- $dcp]"
puts $f "UTC: [clock format [clock seconds] -gmt 1 -format {%Y-%m-%dT%H:%M:%SZ}]"
puts $f "Checkpoint is read only in this flow; no write_checkpoint or implementation command is invoked."
puts $f "Power is a tool estimate using activity/environment retained in the checkpoint and tool defaults where absent. No measured activity file is applied. Inspect report_power confidence and assumptions."
puts $f "Clock periods describe the implemented timing model, not a hardware frequency measurement or a measured maximum frequency."
close $f
report_route_status -file [file join $out route_status.rpt]
report_utilization -file [file join $out utilization.rpt]
report_utilization -hierarchical -file [file join $out utilization_hierarchical.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file [file join $out timing_summary.rpt]
report_clocks -file [file join $out clocks.rpt]
report_clock_utilization -file [file join $out clock_utilization.rpt]
report_power -file [file join $out power.rpt]
report_methodology -file [file join $out methodology.rpt]
report_drc -file [file join $out drc.rpt]
set f [open [file join $out drc_severity_counts.txt] w]
foreach severity {Error {Critical Warning} Warning Advisory} {
    set count 0
    foreach v [get_drc_violations -quiet] {
        if {[get_property SEVERITY $v] eq $severity} {incr count}
    }
    puts $f "$severity: $count"
}
puts $f "HDPR-5 effective severity: [get_property SEVERITY [get_drc_checks HDPR-5]]"
close $f
report_property [get_drc_checks HDPR-5] -file [file join $out hdpr5_properties.txt]
close_design
puts "Reports saved in $out; original checkpoint preserved at $dcp"
exit
