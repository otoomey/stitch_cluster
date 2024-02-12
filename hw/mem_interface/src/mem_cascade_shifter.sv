module mem_cascade_shifter #(
    parameter int unsigned DataWidth     = 0,
    parameter int unsigned NrPorts       = 32'd8,
    parameter int unsigned MemoryLatency = 1,
    /// Request type
    parameter type mem_req_t             = logic,
    /// Response type
    parameter type mem_rsp_t             = logic
) (
    input  logic                       clk_i,
    input  logic                       rst_ni,
    // Inputs
    input  mem_req_t [NrPorts-1:0]     in_req_i,
    output mem_rsp_t [NrPorts-1:0]     in_rsp_o,
    // Multiplexed output.
    output mem_req_t [NrPorts-1:0]     out_req_o,
    input  mem_rsp_t [NrPorts-1:0]     out_rsp_i,
    // Shift amount
    input  logic [$clog2(NrPorts)-1:0] sel_wide_i
);

    

endmodule;

module mem_cascade_shifter #(
    // number of objects in the shifter
    parameter int unsigned Depth = 32'd8,
    parameter type         dtype = logic,
    // minimum amount that an object can be shifted by
    parameter int          MinShift = 1,
    /// Derived parameter *Do not override*
    parameter int unsigned ShiftAmntWidth = $clog2(Depth),
    parameter int unsigned ShiftAmnt = Depth / 2
) (
    input  dtype [Depth-1:0] data_i,
    // *note* if MinShift > 1 then some LSBs of shift_amnt may be ignored
    input  logic [ShiftAmntWidth-1:0] shift_amnt,
    output dtype [Depth-1:0] data_o
);
    dtype [Depth-1:0] data_shifted;
    dtype [Depth-1:0] data_muxd;

    if (ShiftAmnt < MinShift) begin: gen_pass_through
        // don't shift; wire from input to output
        assign data_o = data_i;
    end else begin : gen_shifter
        for (genvar i = 0; i < Depth; i++) begin
            // shift the data by ShiftAmnt
            assign data_shifted[(i+ShiftAmnt) % Depth] = data[i];
        end
        // pick the shifted or unshifted word
        assign data_muxd = shift_amount[$bits(shift_amnt)-1] ? data_shifted : data;
        
        // recurse over next shift level
        cascade_shifter #(
            .Depth(Depth),
            .dtype(dtype),
            .MinShift(MinShift),
            .ShiftAmntWidth(ShiftAmntWidth - 1),
            .ShiftAmnt(ShiftAmnt / 2)
        ) i_cascade_shifter (
            .data_i(data_muxd),
            .shift_amnt(shift_amnt[$bits(shift_amnt)-2:0]),
            .data_o(data_o)
        );
    end
endmodule;