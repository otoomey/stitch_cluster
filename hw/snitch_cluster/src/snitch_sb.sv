module snitch_sb import snitch_pkg::*; #(
    parameter int unsigned AddrWidth,
    parameter int unsigned Depth,
    parameter int unsigned NumTestAddrs,
    // DO NOT OVERWRITE THIS PARAMETER
    parameter type addr_t = logic [AddrWidth-1:0]
) (
    input  logic                        clk_i,
    input  logic                        rst_i,
    input  addr_t                       push_rd_addr_i,
    input  logic                        push_valid_i,
    output logic [Depth-1:0]            entry_index_o,

    input  logic  [Depth-1:0]           pop_index_i,
    input  logic                        pop_valid_i,

    input  addr_t [NumTestAddrs-1:0]    test_addr_i,
    output logic  [NumTestAddrs-1:0]    test_addr_present_o,
    output logic                        full_o,
    output fpu_sb_trace_port_t          trace_port_o
);

    addr_t [Depth-1:0] scoreboard;
    logic [Depth-1:0] occupied;
    logic [Depth-1:0] next_index;
    logic [NumTestAddrs-1:0][Depth-1:0] test_checks;

    // keep track of which indices in the scoreboard are free
    snitch_sb_ipool #(
        .Depth(Depth)
    ) i_indices (
        .clk_i,
        .rst_ni (~rst_i),
        .full_o (/* open */),
        .empty_o(full_o), // no entries left
        .usage_o( /* open */ ),
        .data_i (pop_index_i),
        .push_i (pop_valid_i),
        .data_o (next_index),
        .pop_i  (push_valid_i)
    );

    // write logic
    always_ff @(posedge clk_i) begin
        if (full_o && push_valid_i) begin
            $display("Warning! attempted to push when full: %d->%b", push_rd_addr_i, next_index);
        end
        if (i_indices.usage_o == Depth & pop_valid_i) begin
            $display("Warning! attempted to pop when empty: %b", pop_index_i);
        end
        for (int unsigned i = 0; i < Depth; i++) begin
            if (push_valid_i & next_index[i]) begin
                if (occupied[i]) begin
                    $display("Warning! attempted to push to occupied address");
                end
                // $display("Pushing %d to index %d", push_rd_addr_i, i);
                scoreboard[i] <= push_rd_addr_i;
                occupied[i] <= 1;
            end
            if (pop_valid_i & pop_index_i[i]) begin
                if (occupied[i] != 1) begin
                    $display("Warning! attempted to clear free address");
                end
                occupied[i] <= 0;
                // $display("Popping index %d", i);
            end
            // $display("test checks %d: %x", i, test_checks[i]);
            // $display("mem %d: %x", i, scoreboard[i]);
            // $display("occ %d: %x", i, occupied[i]);
        end
    end

    // test logic
    for (genvar j = 0; j < NumTestAddrs; j++) begin
        for (genvar i = 0; i < Depth; i++) begin
            assign test_checks[j][i] = (scoreboard[i] == test_addr_i[j]) & occupied[i];
        end
    end

    // combine individual checks
    for (genvar j = 0; j < NumTestAddrs; j++) begin
        assign test_addr_present_o[j] = |(test_checks[j]);
    end
    assign entry_index_o = next_index;
    assign trace_port_o.source    = snitch_pkg::SrcFpuSB;
    assign trace_port_o.push_rd_addr_i = push_rd_addr_i;
    assign trace_port_o.push_valid_i = push_valid_i;
    assign trace_port_o.entry_index_o = entry_index_o;
    assign trace_port_o.pop_index_i = pop_index_i;
    assign trace_port_o.pop_valid_i = pop_valid_i;
    //assign trace_port_o.pop_valid_i = test_checks;
    assign trace_port_o.test_addr_i = test_addr_i;
    assign trace_port_o.test_addr_present_o = test_addr_present_o;
    assign trace_port_o.full_o = full_o;

endmodule;