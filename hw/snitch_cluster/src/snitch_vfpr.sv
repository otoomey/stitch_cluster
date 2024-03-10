module snitch_vfpr import snitch_pkg::*; #(
    parameter int unsigned DataWidth = 0,
    parameter int unsigned AddrWidth = 0,
    parameter int unsigned TCDMMemAddrWidth = 0,
    parameter int unsigned RspBufferDepth = 3,
    parameter type tcdm_req_t = logic,
    parameter type tcdm_rsp_t = logic,
    parameter type tcdm_user_t = logic,
    parameter type mem_req_t = logic,
    parameter type mem_rsp_t = logic,
    parameter type tag_t = logic,
    // derived parameters
    parameter type addr_t = logic [AddrWidth-1:0],
    parameter type data_t = logic [DataWidth-1:0]
) (
    input  logic           clk_i,
    input  logic           rst_i,

    // read port
    input  addr_t [2:0]    raddr_i,
    input  logic  [2:0]    ren_i,
    input  logic           rvalid_i,
    output logic           rready_o,
    output data_t [2:0]    rdata_o,
    output logic           rvalid_o,
    input  logic           rready_i,

    // tag to propagate while reading
    input  tag_t           rtag_i,
    output tag_t           rtag_o,

    // write port
    input  tcdm_req_t      wr_req_i,
    output tcdm_rsp_t      wr_rsp_o,

    // memory port
    output mem_req_t [3:0] mem_req_o,
    input  mem_rsp_t [3:0] mem_rsp_i,
    output fpu_vfpr_trace_port_t vfpr_tracer_port
);

    tcdm_req_t [3:0] vfpr_req;
    tcdm_rsp_t [3:0] vfpr_rsp;

    logic [2:0] rvalid_fork, rready_fork;
    logic [2:0] rrsp_valid, rrsp_ready;

    logic spill_valid, spill_ready;
    logic spill_valid_q, spill_ready_q;

    logic [3:0] fork_out_valid, fork_out_ready;
    assign fork_out_valid = {rvalid_fork, spill_valid};
    assign fork_out_ready = {rready_fork, spill_ready};

    stream_fork #(
        .N_OUP(4)
    ) i_fork (
        .clk_i,
        .rst_ni(~rst_i),
        .valid_i(rvalid_i),
        .ready_o(rready_o),
        .valid_o({rvalid_fork, spill_valid}),
        .ready_i({rready_fork, spill_ready})
    );

    for (genvar i = 0; i < 3; i++) begin

        logic ic_in_valid, ic_in_ready;
        logic track_in_valid, track_in_ready;
        
        stream_fork #(
            .N_OUP(2)
        ) i_tcdm_bypass (
            .clk_i,
            .rst_ni(~rst_i),
            .valid_i(rvalid_fork[i]),
            .ready_o(rready_fork[i]),
            .valid_o({ic_in_valid, track_in_valid}),
            .ready_i({ic_in_ready, track_in_ready})
        );

        // bypass requires one cycle of delay to allow
        // ic a change to execute memory request
        logic track_out_valid, track_out_ready;
        logic wait_for_tcdm_rsp;
        spill_register i_track (
            .clk_i,
            .rst_ni(~rst_i),
            .valid_i(track_in_valid),
            .ready_o(track_in_ready),
            .data_i(ren_i[i]),
            .valid_o(track_out_valid),
            .ready_i(track_out_ready),
            .data_o(wait_for_tcdm_rsp)
        );

        logic ic_in_en_valid, ic_in_en_ready;
        stream_filter i_skip_ic (
            .valid_i(ic_in_valid),
            .ready_o(ic_in_ready),
            .drop_i(~ren_i[i]),
            .valid_o(ic_in_en_valid),
            .ready_i(ic_in_en_ready)
        );

        logic cong_out_valid, cong_out_ready;
        logic [1:0] rsp_congestion;
        stream_stall i_full_stall (
            .valid_i(ic_in_en_valid),
            .ready_o(ic_in_en_ready),
            .stall(rsp_congestion >= (RspBufferDepth - 1)),
            .valid_o(cong_out_valid),
            .ready_i(cong_out_ready)
        );

        assign vfpr_req[i].q.addr = raddr_i[i];
        assign vfpr_req[i].q.write = '0;
        assign vfpr_req[i].q.amo = reqrsp_pkg::AMONone;
        assign vfpr_req[i].q.data = '0;
        assign vfpr_req[i].q.strb = '1;
        assign vfpr_req[i].q.user = '0;
        assign vfpr_req[i].q_valid = cong_out_valid;
        assign cong_out_ready = vfpr_rsp[i].q_ready;

        // buffer the interconnect output - necessary because
        // the ic expects output to be always ready
        logic ic_out_valid, ic_out_ready;
        stream_fifo #(
            .FALL_THROUGH ( 1'b1                ),
            .DEPTH        ( RspBufferDepth      ),
            .T            ( data_t              )
        ) i_rsp_buffer (
            .clk_i,
            .rst_ni (~rst_i),
            .flush_i (1'b0),
            .testmode_i(1'b0),
            .usage_o (rsp_congestion),
            .data_i (vfpr_rsp[i].p.data),
            .valid_i (vfpr_rsp[i].p_valid),
            .ready_o (/* open */),
            .data_o (rdata_o[i]),
            .valid_o (ic_out_valid),
            .ready_i (ic_out_ready)
        );

        stream_merge #(
            .N_INP(2)
        ) i_rsp_join (
            .inp_valid_i({ic_out_valid, track_out_valid}),
            .inp_ready_o({ic_out_ready, track_out_ready}),
            .sel_i({wait_for_tcdm_rsp, 1'b1}),
            .oup_valid_o(rrsp_valid[i]),
            .oup_ready_i(rrsp_ready[i])
        );
    end

    assign vfpr_req[3] = wr_req_i;
    assign wr_rsp_o = vfpr_rsp[3];

    snitch_tcdm_interconnect #(
        .NumInp (4),
        .NumOut (4),
        .tcdm_req_t(tcdm_req_t),
        .tcdm_rsp_t(tcdm_rsp_t),
        .mem_req_t(mem_req_t),
        .mem_rsp_t(mem_rsp_t),
        .MemAddrWidth(TCDMMemAddrWidth),
        .DataWidth(DataWidth),
        .user_t(tcdm_user_t)
    ) i_vregfile (
        .clk_i,
        .rst_ni(~rst_i),
        .req_i(vfpr_req),
        .rsp_o(vfpr_rsp),
        .mem_req_o(mem_req_o),
        .mem_rsp_i(mem_rsp_i)
    );

    // buffer the input tag 
    spill_register #(
        .T(tag_t)
    ) i_tag_buffer (
        .clk_i,
        .rst_ni(~rst_i),
        .valid_i(spill_valid),
        .ready_o(spill_ready),
        .data_i(rtag_i),
        .valid_o(spill_valid_q),
        .ready_i(spill_ready_q),
        .data_o(rtag_o)
    );

    // wait for all streams to complete
    stream_join #(
        .N_INP(4)
    ) i_join (
        .inp_valid_i({rrsp_valid, spill_valid_q}),
        .inp_ready_o({rrsp_ready, spill_ready_q}),
        .oup_valid_o(rvalid_o),
        .oup_ready_i(rready_i)
    );

    // always @(posedge clk_i) begin
    //     // if a transactions is happening
    //     if (wr_req_i.q_valid & wr_rsp_o.q_ready & mem_req_o[0].q_valid) begin
    //         $display("- ic: [%x]->[0:%x]", wr_req_i.q.addr, mem_req_o[0].q.addr);
    //     end
    //     if (wr_req_i.q_valid & wr_rsp_o.q_ready & mem_req_o[1].q_valid) begin
    //         $display("- ic: [%x]->[1:%x]", wr_req_i.q.addr, mem_req_o[1].q.addr);
    //     end
    //     if (wr_req_i.q_valid & wr_rsp_o.q_ready & mem_req_o[2].q_valid) begin
    //         $display("- ic: [%x]->[2:%x]", wr_req_i.q.addr, mem_req_o[2].q.addr);
    //     end
    //     if (wr_req_i.q_valid & wr_rsp_o.q_ready & mem_req_o[3].q_valid) begin
    //         $display("- ic: [%x]->[3:%x]", wr_req_i.q.addr, mem_req_o[3].q.addr);
    //     end
    //     if (wr_rsp_o.p_valid) begin
    //         $display("- ic response data:", wr_rsp_o.p.data);
    //     end
    //     if (spill_valid != rvalid_o) begin
    //         $display("tag spill valid state does not match ic state: %b!=%b", spill_valid, rvalid_o);
    //     end
    //     if (spill_ready != rready_o) begin
    //         $display("tag spill ready state does not match ic state: %b!=%b", spill_ready, rready_o);
    //     end
    // end

    assign vfpr_tracer_port.source = SrcFpuVFPR;
    assign vfpr_tracer_port.read = rvalid_i & rready_o;
    assign vfpr_tracer_port.read_result = {vfpr_rsp[0].p_valid, vfpr_rsp[1].p_valid, vfpr_rsp[2].p_valid};
    assign vfpr_tracer_port.reg0 = raddr_i[0];
    assign vfpr_tracer_port.reg1 = raddr_i[1];
    assign vfpr_tracer_port.reg2 = raddr_i[2];
    assign vfpr_tracer_port.reg_enabled = ren_i;
    assign vfpr_tracer_port.data0 = rdata_o[0];
    assign vfpr_tracer_port.data1 = rdata_o[1];
    assign vfpr_tracer_port.data2 = rdata_o[2];
    assign vfpr_tracer_port.write = wr_req_i.q_valid & wr_rsp_o.q_ready;
    assign vfpr_tracer_port.wr_addr = wr_req_i.q.addr;
    assign vfpr_tracer_port.wr_data = wr_req_i.q.data;

endmodule