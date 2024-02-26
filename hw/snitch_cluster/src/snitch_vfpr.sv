module snitch_vfpr #(
    parameter int unsigned DataWidth = 0,
    parameter int unsigned TCDMMemAddrWidth = 0,
    parameter type tcdm_req_t = logic,
    parameter type tcdm_rsp_t = logic,
    parameter type mem_req_t = logic,
    parameter type mem_rsp_t = logic
) (
    input  logic                       clk_i,
    input  logic                       rst_i,
    input  tcdm_req_t                  wr_req_i,
    output tcdm_rsp_t                  wr_rsp_o,
    output mem_req_t [3:0]             mem_req_o,
    input  mem_rsp_t [3:0]             mem_rsp_i
);

    snitch_tcdm_interconnect #(
        .NumInp (1),
        .NumOut (4),
        .tcdm_req_t(tcdm_req_t),
        .tcdm_rsp_t(tcdm_rsp_t),
        .mem_req_t(mem_req_t),
        .mem_rsp_t(mem_rsp_t),
        .MemAddrWidth(TCDMMemAddrWidth),
        .DataWidth(DataWidth)
    ) i_vregfile (
        .clk_i,
        .rst_ni(~rst_i),
        .req_i(wr_req_i),
        .rsp_o(wr_rsp_o),
        .mem_req_o(mem_req_o),
        .mem_rsp_i(mem_rsp_i)
    );

    initial begin
        $display("vfpr mem addr width %d, data width %d", TCDMMemAddrWidth, DataWidth);
        $display("mem req addr width: %d", $bits(mem_req_o[0].q.addr));
    end

    always @(posedge clk_i) begin
        // if a transactions is happening
        if (wr_req_i.q_valid & wr_rsp_o.q_ready & mem_req_o[0].q_valid) begin
            $display("- ic: [%x]->[0:%x]", wr_req_i.q.addr, mem_req_o[0].q.addr);
        end
        if (wr_req_i.q_valid & wr_rsp_o.q_ready & mem_req_o[1].q_valid) begin
            $display("- ic: [%x]->[1:%x]", wr_req_i.q.addr, mem_req_o[1].q.addr);
        end
        if (wr_req_i.q_valid & wr_rsp_o.q_ready & mem_req_o[2].q_valid) begin
            $display("- ic: [%x]->[2:%x]", wr_req_i.q.addr, mem_req_o[2].q.addr);
        end
        if (wr_req_i.q_valid & wr_rsp_o.q_ready & mem_req_o[3].q_valid) begin
            $display("- ic: [%x]->[3:%x]", wr_req_i.q.addr, mem_req_o[3].q.addr);
        end
        if (wr_rsp_o.p_valid) begin
            $display("- ic response data:", wr_rsp_o.p.data);
        end
    end

endmodule