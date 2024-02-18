module stitch_sb #(
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
    output logic                        entry_index_o,

    input  logic  [Depth-1:0]           pop_index_i,
    input  logic                        pop_valid_i,

    input  addr_t [NumTestAddrs-1:0]    test_addr_i,
    output logic  [NumTestAddrs-1:0]    test_addr_present_o,
    output logic                        full_o
);

    logic [Depth-1:0][AddrWidth-1:0] scoreboard;
    logic [Depth-1:0] next_index;

    // keep track of which indices in the scoreboard are free
    stitch_sb_ipool #(
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
    always_ff @(posedge clk) begin
        for (int unsigned i = 0; i < Depth; i++) begin
            if (push_valid_i & (next_index == i)) begin
                scoreboard[i] = push_rd_addr_i;
            end
        end
    end

    // test logic
    logic [NumTestAddrs-1:0][Depth-1:0] test_checks;
    for (genvar j = 0; j < NumTestAddrs; j++) begin
        for (genvar i = 0; i < Depth; i++) begin
            assign test_checks[j][i] = (scoreboard[i] == test_addr_i[j]);
        end
    end

    // combine individual checks
    for (genvar j = 0; j < NumTestAddrs; j++) begin
        assign test_addr_present_o[j] = |(test_checks[j]);
    end
    assign entry_index_o = next_index;

endmodule;
