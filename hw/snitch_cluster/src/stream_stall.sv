module stream_stall (
    input logic valid_i,
    output logic ready_o,
    input logic stall,
    output logic valid_o,
    input logic ready_i
);
    always_comb begin
        valid_o = valid_i;
        ready_o = ready_i;
        if (stall) begin
            valid_o = 0;
            ready_o = 0;
        end
    end
endmodule