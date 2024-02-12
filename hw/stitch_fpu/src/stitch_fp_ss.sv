// The stitch fpu subsystem replaces the snitch variant
// The most notable differences are:
// - The FPU register files are memory mapped
// - A much more tightly coupled interface
// - The use of an array of FPUs controlled by a single
//   core

// The following bits and pieces are required to make this work.
// First, the memory interface structure is modified to arbitrate
// bank level access between FPU units and the general system bus.
// The FPUs use a modifed scoreboard that keeps track of rd register
// addresses. If a read is detected to a marked address, the FPU is
// stalled until the write is ready.

// This is accompished using a ringbuffer where each address is matched
// against the desired read addresses. If any match, a stall flag is
// triggered

// The FPU also contains 4 address generators. These store the current
// memory offset added to each register read in the 4 banks:
// Bank 1: f[0,1,2,3,4,5,6,7]
// Bank 2: f[8,9,10,11,12,13,14,15]
// Bank 3: f[16,17,18,19,20,21,22,23]
// Bank 4: f[24,25,26,27,28,29,30,31]
// This allows frep staggering to easily target all regs
// The address generators are configured as follows:
//  start -  starting offset
//  stride - how much to increment the offset by
// The stride is added after every frep once all staggers are done

// The FPU contains a bitmask CSR onehot encoded for each FPU
// this is used to enable or disable certain FPUs, as well as to enable
// or disabled writing configuration of those FPUs

// The FPU contains a configurable number of LSUs. These are connected to
// the ic tcdm interconnect

// there is a potential data race between "normal" LSU operations and 
// in memory FPU computation

// in general all instructions are executed across all FPUs
// in the case of writeback operations to the CPU, FPU 0 has precedence
// if the rd registers are staggered, then the FPUs write to staggered
// offsets. This deviates from the normal datapath for frep.

// outside frep FPUs are synchronised with integer pipeline instructions,
// so the programmer can optionally choose to enable or disabe certain
// units based on these branch conditions

// for this reason, the result of FLT and FLE is written onehot-encoded
// to rd when staggering is disabled. This gives a nice way to quickly enable
// or disable certain paths based on a result.

// When an FPU is disabled, its read paths are disabled. Any values that still
// need to be written are written (this should probably never happen - enable/disable
// requires pipeline synhcronization)