module stitch_vfpr import snitch_pkg::*; #(
    parameter int unsigned DataWidth = 0,
    parameter int unsigned AddrWidth = 0,
    parameter type tag_t = logic,
    parameter type vfpr_req_t = logic,
    parameter type vfpr_rsp_t = logic,
    parameter type dbankreq_t = logic,
    parameter type dbankrsp_t = logic
) (
    input  logic                        clk_i,
    input  logic                        rst_i,
    // the addresses to read the registers from
    input  logic [2:0][AddrWidth-1:0]   raddr_i,
    input tag_t                         rtag_i,
    // enable/disable different reads
    input  logic [2:0]                  ren_i,
    // input is valid
    input  logic                        rvalid_i,
    // ready to read another set of addresses
    output logic                        rready_o,

    input vfpr_req_t                    wr_port_req_i,
    output vfpr_req_t                   wr_port_rsp_o,

    // register data output
    output logic [2:0][DataWidth-1:0]   rdata_o,
    output tag_t                        rtag_o,
    // indicates all outputs are valid
    output logic                        rdata_valid_o,
    // output ready to receive next set of registers
    input  logic                        rdata_ready_i,

    output dbankreq_t [3:0]             mem_req_o,
    input  dbankrsp_t [3:0]             mem_rsp_i
);

    // typedef logic [AddrWidth-1:0]           addr_t;
    // typedef logic [DataWidth-1:0]           data_t;
    // typedef logic [DataWidth/8-1:0]         strb_t;
    // `TCDM_TYPEDEF_ALL(vfpr, addr_t, data_t, strb_t, logic)

    vfpr_req_t [3:0] vfpr_reqs;
    vfpr_rsp_t [3:0] vfpr_rsps;

    logic [2:0] rvalid_fork, rready_fork;
    logic [2:0] rvalid_bypass, rready_bypass;
    logic [2:0] rvalid_bypass_q, rready_bypass_q;

    stream_fork #(
        .N_OUP(3)
    ) i_fork (
        .clk_i,
        .rst_ni(~rst_i),
        .valid_i(rvalid_i),
        .ready_o(rready_o),
        .valid_o(rvalid_fork),
        .ready_i(rready_fork)
    );

    logic [2:0] rrsp_valid, rrsp_ready;

    for (genvar i = 0; i < 3; i++) begin
        // bypass the ic if this particular ready should
        // not take place
        stream_demux #(
            .N_OUP(2)
        ) i_tcdm_bypass (
            .inp_valid_i(rvalid_fork[i]),
            .inp_ready_o(rready_fork[i]),
            .oup_sel_i(ren_i[i]),
            .oup_valid_o(rvalid_bypass[i]),
            .oup_ready_i(rready_bypass[i])
        );

        // bypass requires one cycle of delay to allow
        // ic a change to execute memory request
        spill_register i_bypass (
            .clk_i,
            .rst_ni(~rst_i),
            .valid_i(rvalid_bypass[i]),
            .ready_o(rready_bypass[i]),
            .data_i('x),
            .valid_o(rvalid_bypass_q[i]),
            .ready_i(rready_bypass_q[i]),
            .data_o(/*unused*/)
        );

        assign vfpr_reqs[i].q.addr = raddr_i[i];
        assign vfpr_reqs[i].q.write = '0;
        assign vfpr_reqs[i].q.amo = reqrsp_pkg::AMONone;
        assign vfpr_reqs[i].q.data = '0;
        assign vfpr_reqs[i].q.strb = '1;
        assign vfpr_reqs[i].q.user = '0;
        assign vfpr_reqs[i].q_valid = rvalid_fork[i];
        assign vfpr_rsps[i].q_ready = rready_fork[i];
    end

    assign vfpr_reqs[3] = wr_port_req_i;
    assign vfpr_rsps[3] = wr_port_rsp_o;

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

    for (genvar i = 0; i < 3; i++) begin
        logic ic_valid, ic_ready;

        // buffer the interconnect output - necessary because
        // the ic expects output to be always ready
        fall_through_register #(
            .T(data_t)
        ) i_rsp_buffer (
            .clk_i,
            .rst_ni(~rst_i),
            .clr_i('0),
            .testmode_i('0),
            .valid_i(vfpr_rsps[i].p_valid),
            .ready_o(/* unused */),
            .data_i(vfpr_rsps[i].p.data),
            .valid_o(ic_valid),
            .ready_i(ic_ready),
            .data_o(rdata_o)
        );

        // combine bypass and ic stream, choosing latter
        // with priority
        stream_reduce #(
            .data_t
        ) i_reduce (
            .valid_i({ic_valid, rvalid_bypass_q[i]}),
            .ready_o({ic_ready, rready_bypass_q[i]}),
            .valid_o(rrsp_valid[i]),
            .ready_i(rrsp_ready[i])
        );
    end

    // wait for all streams to complete
    stream_join #(
        .N_INP(3)
    ) i_join (
        .inp_valid_i(rrsp_valid),
        .inp_ready_o(rrsp_ready),
        .oup_valid_o(rdata_valid_o),
        .oup_ready_i(rdata_ready_i)
    );

    // buffer the input tag 
    spill_register #(
        .T(tag_t)
    ) i_tag_buffer (
        .clk_i,
        .rst_ni(~rst_i),
        .valid_i(rvalid_i),
        .ready_o(rready_o),
        .data_i(rtag_i),
        .valid_o(rdata_valid_o),
        .ready_i(rdata_ready_i),
        .data_o(rtag_o)
    );

endmodule