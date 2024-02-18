module stitch_vfpr import snitch_pkg::*; #(
    parameter int unsigned DataWidth = 0,
    parameter int unsigned AddrWidth = 0,
    parameter type tag_t = logic,
    parameter type dbankreq_t = logic,
    parameter type dbankrsp_t = logic
) (
    input  logic                        clk_i,
    input  logic                        rst_i,
    // the addresses to read the registers from
    input  logic [2:0][AddrWidth-1:0]   raddr_i,
    // which addresses are valid
    input  logic [2:0]                  rvalid_i,
    // ready to read another set of addresses
    output logic [2:0]                  rready_o,

    // address to write to
    input  logic [AddrWidth-1:0]        waddr_i,
    // the data to write to address
    input  logic [DataWidth-1:0]        wdata_i,
    // write data is valid
    input  logic                        wvalid_i,
    // ready to write
    output logic                        wready_o,

    // register data output
    output logic [2:0][DataWidth-1:0]   rdata_o,
    // indicates all outputs are valid
    output logic                        rdata_valid_o,
    // output ready to receive next set of registers
    input  logic                        rdata_ready_i,

    output dbankreq_t [3:0]             mem_req_o,
    input  dbankrsp_t [3:0]             mem_rsp_i
);

    typedef logic [AddrWidth-1:0]           addr_t;
    typedef logic [DataWidth-1:0]           data_t;
    typedef logic [DataWidth/8-1:0]         strb_t;
    `TCDM_TYPEDEF_ALL(vfpr, addr_t, data_t, strb_t, logic)

    vfpr_req_t [3:0] vfpr_reqs;
    vfpr_rsp_t [3:0] vfpr_rsps;

    logic rsp_ready;

    // Request Handler

    for (genvar i = 0; i < 3; i++) begin
        assign vfpr_reqs[i].q.addr = raddr_i[i];
        assign vfpr_reqs[i].q.write = '0;
        assign vfpr_reqs[i].q.amo = reqrsp_pkg::AMONone;
        assign vfpr_reqs[i].q.data = '0;
        assign vfpr_reqs[i].q.strb = '1;
        assign vfpr_reqs[i].q.user = '0;

        // if the input is valid, and interface is ready, initiate
        assign vfpr_reqs[i].q_valid = rvalid_i[i] & vfpr_rsps[i].q_ready;
        assign rready_o[i] = vfpr_rsps[i].q_ready & rsp_ready;
    end

    assign vfpr_reqs[3].q.addr = waddr_i;
    assign vfpr_reqs[3].q.write = '1;
    assign vfpr_reqs[3].q.amo = reqrsp_pkg::AMONone;
    assign vfpr_reqs[3].q.data = wdata_i;
    assign vfpr_reqs[3].q.strb = '1;
    assign vfpr_reqs[3].q.user = '0;

    // if the input is valid, and interface is ready, initiate
    assign vfpr_reqs[3].q_valid = wvalid_i & vfpr_rsps[3].q_ready;
    assign wready_o = vfpr_rsps[3].q_ready & rsp_ready;

    snitch_tcdm_interconnect #(
        .NumInp (4),
        .NumOut (4),
        .tcdm_req_t,
        .tcdm_rsp_t,
        .mem_req_t(dbankreq_t),
        .mem_rsp_t(dbankrsp_t),
        .MemAddrWidth(BankAddrWidth),
        .DataWidth(FLEN)
    ) i_vregfile (
        .clk_i,
        .rst_ni(~rst_i),
        .req_i(vfpr_reqs),
        .rsp_o(vfpr_rsps),
        .mem_req_o(mem_req_o),
        .mem_rsp_i(mem_rsp_i)
    );

    typedef struct packed {
        logic [2:0] rvalid;
    } rsp_buf_t;

    // Response Handler
    // data can arrive in any order and must be accepted immediately
    data_t [2:0] reg_results_q;
    logic [2:0] reg_results_valid_q;
    rsp_buf_t rsp_buf_q, rsp_buf;

    assign rsp_buf.rvalid = rvalid_i;

    `FFLAR(reg_results_q[0], vfpr_rsps.p.data, vfpr_rsps[0].p_valid, '0, clk_i, rst_i)
    `FFLAR(reg_results_valid_q[0], vfpr_rsps.p.data, vfpr_rsps[0].p_valid, '0, clk_i, rst_i)
    `FFLAR(reg_results_q[1], vfpr_rsps.p.data, vfpr_rsps[1].p_valid, '0, clk_i, rst_i)
    `FFLAR(reg_results_valid_q[1], vfpr_rsps.p.data, vfpr_rsps[1].p_valid, '0, clk_i, rst_i)
    `FFLAR(reg_results_q[2], vfpr_rsps.p.data, vfpr_rsps[2].p_valid, '0, clk_i, rst_i)
    `FFLAR(reg_results_valid_q[2], vfpr_rsps.p.data, vfpr_rsps[2].p_valid, '0, clk_i, rst_i)

    logic req_valid;
    spill_register  #(
        .T      ( rsp_buf_t ),
    ) i_spill_register_acc (
        .clk_i   ,
        .rst_ni  ( ~rst_i ),
        .valid_i ( vfpr_reqs[0].q_valid & vfpr_reqs[1].q_valid & vfpr_reqs[2].q_valid ),
        .ready_o ( rsp_ready ),
        .data_i  ( rsp_buf ),
        .valid_o ( req_valid ),
        .ready_i ( rdata_ready_i ),
        .data_o  ( rsp_buf_q )
    );

    assign rdata_o = reg_results_q;
    assign rdata_valid_o = req_valid & &(~rsp_buf_q.rvalid | reg_results_valid_q);

endmodule