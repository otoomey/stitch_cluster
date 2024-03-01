// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "common_cells/registers.svh"
`include "common_cells/assertions.svh"

// Floating Point Subsystem
module snitch_fp_ss import snitch_pkg::*; #(
  parameter int unsigned AddrWidth = 32,
  parameter int unsigned TCDMMemAddrWidth = 32,
  parameter int unsigned DataWidth = 32,
  parameter int unsigned NumFPOutstandingLoads = 0,
  parameter int unsigned NumFPOutstandingMem = 0,
  parameter int unsigned NumFPUSequencerInstr = 0,
  parameter type dreq_t = logic,
  parameter type drsp_t = logic,
  parameter type mem_req_t = logic,
  parameter type mem_rsp_t = logic,
  parameter type tcdm_user_t = logic,
  parameter type tcdm_req_t = logic,
  parameter type tcdm_rsp_t = logic,
  parameter bit RegisterSequencer = 0,
  parameter bit RegisterFPUIn     = 0,
  parameter bit RegisterFPUOut    = 0,
  parameter bit Xfrep = 1,
  parameter fpnew_pkg::fpu_implementation_t FPUImplementation = '0,
  parameter bit Xssr = 1,
  parameter int unsigned NumSsrs = 0,
  parameter logic [NumSsrs-1:0][4:0]  SsrRegs = '0,
  parameter type acc_req_t = logic,
  parameter type acc_resp_t = logic,
  parameter bit RVF = 1,
  parameter bit RVD = 1,
  parameter bit XF16 = 0,
  parameter bit XF16ALT = 0,
  parameter bit XF8 = 0,
  parameter bit XF8ALT = 0,
  parameter bit XFVEC = 0,
  parameter int unsigned FLEN = DataWidth,
  /// Derived parameter *Do not override*
  parameter type addr_t = logic [AddrWidth-1:0],
  parameter type data_t = logic [DataWidth-1:0],
  parameter int unsigned ReqAddrWidth = TCDMMemAddrWidth,
  parameter int unsigned ReqPrefixWidth = ReqAddrWidth-3,
  parameter type regaddr_t = logic [ReqAddrWidth-1:0]
) (
  input  logic             clk_i,
  input  logic             rst_i,
  // pragma translate_off
  output fpu_trace_port_t  trace_port_o,
  output fpu_sequencer_trace_port_t sequencer_tracer_port_o,
  output fpu_sb_trace_port_t sb_tracer_port_o,
  output fpu_vfpr_trace_port_t vfpr_tracer_port_o,
  // pragma translate_on
  input  logic [31:0]      hart_id_i,
  // Accelerator Interface - Slave
  input  acc_req_t         acc_req_i,
  input  logic             acc_req_valid_i,
  output logic             acc_req_ready_o,
  output acc_resp_t        acc_resp_o,
  output logic             acc_resp_valid_o,
  input  logic             acc_resp_ready_i,
  // TCDM Data Interface for regular FP load/stores.
  output dreq_t            data_req_o,
  input  drsp_t            data_rsp_i,
  // TCDM Data Interface for IC to connect to register bank.
  input  tcdm_req_t        data_req_i,
  output tcdm_rsp_t        data_rsp_o,
  // Memory interfaces to SRAM
  output mem_req_t [3:0]   mem_req_o,
  input  mem_rsp_t [3:0]   mem_rsp_i,
  // Register Interface
  // FPU **un-timed** Side-channel
  input  fpnew_pkg::roundmode_e fpu_rnd_mode_i,
  input  fpnew_pkg::fmt_mode_t  fpu_fmt_mode_i,
  output fpnew_pkg::status_t    fpu_status_o,
  // SSR Interface
  output logic  [2:0][4:0] ssr_raddr_o,
  input  data_t [2:0]      ssr_rdata_i,
  output logic  [2:0]      ssr_rvalid_o,
  input  logic  [2:0]      ssr_rready_i,
  output logic  [2:0]      ssr_rdone_o,
  output logic  [0:0][4:0] ssr_waddr_o,
  output data_t [0:0]      ssr_wdata_o,
  output logic  [0:0]      ssr_wvalid_o,
  input  logic  [0:0]      ssr_wready_i,
  output logic  [0:0]      ssr_wdone_o,
  // SSR stream control interface
  input  logic             streamctl_done_i,
  input  logic             streamctl_valid_i,
  output logic             streamctl_ready_o,
  // Core event strobes
  output core_events_t core_events_o
);

  localparam ScoreboardDepth = 5;

  // fpnew_pkg::operation_e  vfpr_tag_in.fpu_op;
  // fpnew_pkg::roundmode_e  vfpr_tag_in.fpu_rnd_mode;
  // fpnew_pkg::fp_format_e  vfpr_tag_in.src_fmt, vfpr_tag_in.dst_fmt;
  // fpnew_pkg::int_format_e vfpr_tag_in.int_fmt;
  // logic                   vfpr_tag_in.vectorial_op;
  // logic                   vfpr_tag_in.set_dyn_rm;

  // logic [2:0][4:0]      vfpr_tag_in.fpr_raddr;
  logic [2:0][FLEN-1:0] fpr_rdata;

  logic [0:0]           fpr_we;
  logic [0:0][4:0]      fpr_waddr;
  logic [0:0][FLEN-1:0] fpr_wdata;
  logic [0:0]           fpr_wvalid;
  logic [0:0]           fpr_wready;

  logic [ScoreboardDepth-1:0] fpr_windex;
  logic [0:0]                 sb_pop_valid;

  logic ssr_active_d, ssr_active_q, ssr_active_ena;
  `FFLAR(ssr_active_q, Xssr & ssr_active_d, ssr_active_ena, 1'b0, clk_i, rst_i)

  typedef struct packed {
    logic       ssr; // write-back to SSR at rd
    logic       acc; // write-back to result bus
    logic [4:0] rd;  // write-back to floating point regfile
    logic [ScoreboardDepth-1:0] fpr_windex;
  } tag_t;
  tag_t fpu_tag_in, fpu_tag_out;
  tag_t lsu_tag_in, lsu_tag_out;

  // scoreboard
  // logic [ScoreboardDepth-1:0] rd_index;
  logic [3:0][4:0] sb_tests;
  logic [3:0] sb_collision;
  logic sb_full;

  // logic vfpr_tag_in.is_fpu;
  logic [2:0][FLEN-1:0] op;
  logic [2:0] vfpr_op_ready;

  logic        lsu_in_ready;
  logic        lsu_in_valid;
  logic [FLEN-1:0] ld_result;
  logic        lsu_pvalid;
  logic        lsu_pready;
  // logic vfpr_tag_in.is_store, vfpr_tag_in.is_load;

  // logic [31:0] sb_d, sb_q;
  // logic vfpr_tag_in.rd_is_fp;
  // `FFAR(sb_q, sb_d, '0, clk_i, rst_i)

  logic csr_instr;

  // FPU Controller
  logic fpu_out_valid, fpu_out_ready;
  logic fpu_in_valid, fpu_in_ready;

  // WR tcdm requests
  tcdm_req_t fpr_wr_req;
  tcdm_rsp_t fpss_wr_rsp;

  tcdm_req_t vfpr_req;
  tcdm_req_t vfpr_rsp;

  typedef enum logic [2:0] {
    None,
    AccBus,
    RegA, RegB, RegC,
    RegBRep, // Replication for vectors
    RegDest
  } op_select_e;
  // op_select_e [2:0] vfpr_tag_in.op_select;

  typedef enum logic [1:0] {
    ResNone, ResAccBus
  } result_select_e;
  result_select_e result_select;

  // logic vfpr_tag_in.op_mode;

  logic [4:0] rs1, rs2, rs3; // vfpr_tag_in.rd;

  // LSU
  typedef enum logic [1:0] {
    Byte       = 2'b00,
    HalfWord   = 2'b01,
    Word       = 2'b10,
    DoubleWord = 2'b11
  } ls_size_e;
  // ls_size_e vfpr_tag_in.ls_size;


  logic dst_ready;

  // VFPR Controller
  logic vfpr_in_valid, vfpr_in_ready;
  logic vfpr_out_valid, vfpr_out_ready;

  typedef struct packed {
    op_select_e [2:0]       op_select;
    logic [2:0][4:0]        fpr_raddr; // TODO remove
    fpnew_pkg::operation_e  fpu_op;
    fpnew_pkg::roundmode_e  fpu_rnd_mode;
    fpnew_pkg::fp_format_e  src_fmt;
    fpnew_pkg::fp_format_e  dst_fmt;
    fpnew_pkg::int_format_e int_fmt;
    logic                   vectorial_op;
    logic                   set_dyn_rm;
    logic                   op_mode;
    logic                   is_fpu;
    logic                   is_store;
    logic                   is_load;
    logic                   rd_is_fp;
    logic                   rd_is_acc;
    ls_size_e               ls_size;
    data_t                  data_arga;
    data_t                  data_argb;
    data_t                  data_argc;
    logic [4:0]             rd;
    regaddr_t               rd_bnk_addr;
    logic [ScoreboardDepth-1:0] fpr_windex;
  } vfpr_tag_t;
  
  vfpr_tag_t vfpr_tag_in, vfpr_tag_out;

  // -------------
  // FPU Sequencer
  // -------------
  acc_req_t         acc_req, acc_req_q;
  logic             acc_req_valid, acc_req_valid_q;
  logic             acc_req_ready, acc_req_ready_q;
  if (Xfrep) begin : gen_fpu_sequencer
    snitch_sequencer #(
      .AddrWidth (AddrWidth),
      .DataWidth (DataWidth),
      .Depth     (NumFPUSequencerInstr)
    ) i_snitch_fpu_sequencer (
      .clk_i,
      .rst_i,
      // pragma translate_off
      .trace_port_o     ( sequencer_tracer_port_o ),
      // pragma translate_on
      .inp_qaddr_i      ( acc_req_i.addr      ),
      .inp_qid_i        ( acc_req_i.id        ),
      .inp_qdata_op_i   ( acc_req_i.data_op   ),
      .inp_qdata_arga_i ( acc_req_i.data_arga ),
      .inp_qdata_argb_i ( acc_req_i.data_argb ),
      .inp_qdata_argc_i ( acc_req_i.data_argc ),
      .inp_qvalid_i     ( acc_req_valid_i     ),
      .inp_qready_o     ( acc_req_ready_o     ),
      .oup_qaddr_o      ( acc_req.addr        ),
      .oup_qid_o        ( acc_req.id          ),
      .oup_qdata_op_o   ( acc_req.data_op     ),
      .oup_qdata_arga_o ( acc_req.data_arga   ),
      .oup_qdata_argb_o ( acc_req.data_argb   ),
      .oup_qdata_argc_o ( acc_req.data_argc   ),
      .oup_qvalid_o     ( acc_req_valid       ),
      .oup_qready_i     ( acc_req_ready       ),
      .streamctl_done_i,
      .streamctl_valid_i,
      .streamctl_ready_o
    );
  end else begin : gen_no_fpu_sequencer
    // pragma translate_off
    assign sequencer_tracer_port_o = 0;
    // pragma translate_on
    assign acc_req_ready_o = acc_req_ready;
    assign acc_req_valid = acc_req_valid_i;
    assign acc_req = acc_req_i;
  end

  // Optional spill-register
  spill_register  #(
    .T      ( acc_req_t                           ),
    .Bypass ( !RegisterSequencer || !Xfrep )
  ) i_spill_register_acc (
    .clk_i   ,
    .rst_ni  ( ~rst_i          ),
    .valid_i ( acc_req_valid   ),
    .ready_o ( acc_req_ready   ),
    .data_i  ( acc_req         ),
    .valid_o ( acc_req_valid_q ),
    .ready_i ( acc_req_ready_q ),
    .data_o  ( acc_req_q       )
  );

  assign vfpr_tag_in.data_arga = acc_req_q.data_arga;
  assign vfpr_tag_in.data_argb = acc_req_q.data_argb;
  assign vfpr_tag_in.data_argc = acc_req_q.data_argc;
  assign vfpr_tag_in.rd_bnk_addr = vfpr_tag_in.rd; // TODO

  assign sb_tests[0] = vfpr_tag_in.fpr_raddr[0];
  assign sb_tests[1] = vfpr_tag_in.fpr_raddr[1];
  assign sb_tests[2] = vfpr_tag_in.fpr_raddr[2];
  assign sb_tests[3] = vfpr_tag_in.rd;
  // assign sb_tests = {vfpr_tag_in.rd, vfpr_tag_in.fpr_raddr[2], vfpr_tag_in.fpr_raddr[1], vfpr_tag_in.fpr_raddr[0]};

  logic sb_push_valid;
  snitch_sb #(
    .AddrWidth(5),
    .Depth(ScoreboardDepth),
    .NumTestAddrs(4)
  ) i_sb (
    .clk_i,
    .rst_i,
    .push_rd_addr_i(vfpr_tag_in.rd),
    .push_valid_i(sb_push_valid),
    .entry_index_o(vfpr_tag_in.fpr_windex),
    .pop_index_i(fpr_windex),
    .pop_valid_i(sb_pop_valid),
    .test_addr_i(sb_tests),
    .test_addr_present_o(sb_collision),
    .full_o(sb_full),
    .trace_port_o(sb_tracer_port_o)
  );

  // Ensure SSR CSR only written on instruction commit
  assign ssr_active_ena = acc_req_valid_q & acc_req_ready_q;

  // this handles WAW Hazards - Potentially this can be relaxed if necessary
  // at the expense of increased timing pressure
  // assign dst_ready = ~(vfpr_tag_in.rd_is_fp & sb_q[vfpr_tag_in.rd]);
  assign dst_ready = ~(vfpr_tag_in.rd_is_fp & (sb_collision[3] | sb_full));

  // if this is a csr instruction, or something we don't recognize, skip
  logic ex_ins_valid, ex_ins_ready;
  logic drop_ins;
  assign drop_ins = csr_instr | ~(vfpr_tag_in.is_fpu | vfpr_tag_in.is_load | vfpr_tag_in.is_store | (result_select == ResAccBus));
  stream_filter i_csr_filter (
    .valid_i(acc_req_valid_q),
    .ready_o(acc_req_ready_q),
    .drop_i(drop_ins),
    .valid_o(ex_ins_valid),
    .ready_i(ex_ins_ready)
  );

  // stall the stream if rd isn't ready yet
  logic ex_dst_valid, ex_dst_ready;
  stream_stall i_dst_stall (
    .valid_i(ex_ins_valid),
    .ready_o(ex_ins_ready),
    .stall(~dst_ready),
    .valid_o(ex_dst_valid),
    .ready_i(ex_dst_ready)
  );

  // demux accelerator write datapath (despite being unused atm)
  logic vfpr_dp_valid, vfpr_dp_ready;
  logic acc_dp_valid, acc_dp_ready;
  logic acc_demux_sel;
  assign acc_demux_sel = (result_select == ResAccBus);
  stream_demux #(
    .N_OUP(2)
  ) i_acc_demux (
    .inp_valid_i(ex_dst_valid),
    .inp_ready_o(ex_dst_ready),
    .oup_sel_i(acc_demux_sel),
    .oup_valid_o({acc_dp_valid, vfpr_dp_valid}),
    .oup_ready_i({acc_dp_ready, vfpr_dp_ready})
  );

  // stall if operands are not ready
  stream_stall i_vfpr_op_stall (
    .valid_i(vfpr_dp_valid),
    .ready_o(vfpr_dp_ready),
    .stall(~(&vfpr_op_ready)),
    .valid_o(vfpr_in_valid),
    .ready_i(vfpr_in_ready)
  );

  // ready to commit - check in addr to scoreboard
  assign sb_push_valid = (vfpr_in_valid & vfpr_in_ready & vfpr_tag_in.rd_is_fp);

  // determine whether to use lsu or fpu
  // (assumes that op can only be lsu op or fpu op)
  logic fpu_lsu_sel;
  assign fpu_lsu_sel = vfpr_tag_out.is_load | vfpr_tag_out.is_store;
  stream_demux #(
    .N_OUP(2)
  ) i_fpu_lsu_demux (
    .inp_valid_i(vfpr_out_valid),
    .inp_ready_o(vfpr_out_ready),
    .oup_sel_i(fpu_lsu_sel),
    .oup_valid_o({lsu_in_valid, fpu_in_valid}),
    .oup_ready_i({lsu_in_ready, fpu_in_ready})
  );

  always_ff @(posedge clk_i) begin
    if (hart_id_i == 0) begin
      if (ex_ins_valid & ex_ins_ready) begin
        $display("%d: ex_dp", $time);
      end
      if (ex_dst_valid & ex_dst_ready) begin
        $display("%d: ex_dst", $time);
      end
      if (vfpr_dp_valid & vfpr_dp_ready) begin
        $display("%d: vfpr_dp", $time);
      end
      if (acc_dp_valid & acc_dp_ready) begin
        $display("%d: acc_dp", $time);
      end
      if (vfpr_in_valid & vfpr_in_ready) begin
        $display("%d: vfpr_in", $time);
      end
      if (vfpr_out_valid & vfpr_out_ready) begin
        $display("%d: vfpr_out", $time);
      end
      if (sb_push_valid) begin
        $display("%d: sb_push", $time);
      end
      if (fpu_in_valid & fpu_in_ready) begin
        $display("%d: fpu_in", $time);
      end
      if (fpu_out_valid & fpu_out_ready) begin
        $display("%d: fpu_out, %b, %b, %b, %b, %b", 
          $time, fpu_in_valid, vfpr_out_valid, vfpr_tag_out.is_fpu, vfpr_tag_out.is_load, vfpr_tag_out.is_store
        );
      end
      if (lsu_in_valid & lsu_in_ready) begin
        $display("%d: lsu_in", $time);
      end
      if (lsu_pvalid & lsu_pready) begin
        $display("%d: lsu_out", $time);
      end
    end
  end

  // either the FPU or the regfile produced a result
  assign acc_resp_valid_o = (fpu_tag_out.acc & fpu_out_valid);
  // stall FPU if we forward from reg
  assign fpu_out_ready = ((fpu_tag_out.acc & acc_resp_ready_i) | (~fpu_tag_out.acc & fpr_wready));

  // FPU Result
  logic [FLEN-1:0] fpu_result;

  // FPU Tag
  assign acc_resp_o.id = fpu_tag_out.rd;
  // accelerator bus write-port
  assign acc_resp_o.data = fpu_result;

  assign vfpr_tag_in.rd = acc_req_q.data_op[11:7];
  assign rs1 = acc_req_q.data_op[19:15];
  assign rs2 = acc_req_q.data_op[24:20];
  assign rs3 = acc_req_q.data_op[31:27];

  // Scoreboard/Operand Valid
  // Track floating point destination registers
  // always_comb begin
  //   sb_d = sb_q;
  //   // if the instruction is going to write the FPR mark it
  //   if (acc_req_valid_q & acc_req_ready_q & vfpr_tag_in.rd_is_fp) sb_d[vfpr_tag_in.rd] = 1'b1;
  //   // reset the value if we are committing the register
  //   if (fpr_we) sb_d[fpr_waddr] = 1'b0;
  //   // don't track any dependencies for SSRs if enabled
  //   if (ssr_active_q) begin
  //     for (int i = 0; i < NumSsrs; i++) sb_d[SsrRegs[i]] = 1'b0;
  //   end
  // end

  // Determine whether destination register is SSR
  always_comb begin
    acc_resp_o.error = 1'b0;
    vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
    vfpr_tag_in.is_fpu = 1'b1;
    vfpr_tag_in.fpu_rnd_mode = (fpnew_pkg::roundmode_e'(acc_req_q.data_op[14:12]) == fpnew_pkg::DYN)
                   ? fpu_rnd_mode_i
                   : fpnew_pkg::roundmode_e'(acc_req_q.data_op[14:12]);

    vfpr_tag_in.set_dyn_rm = 1'b0;

    vfpr_tag_in.src_fmt = fpnew_pkg::FP32;
    vfpr_tag_in.dst_fmt = fpnew_pkg::FP32;
    vfpr_tag_in.int_fmt = fpnew_pkg::INT32;

    result_select = ResNone;

    vfpr_tag_in.op_select[0] = None;
    vfpr_tag_in.op_select[1] = None;
    vfpr_tag_in.op_select[2] = None;

    vfpr_tag_in.vectorial_op = 1'b0;
    vfpr_tag_in.op_mode = 1'b0;

    fpu_tag_in.rd = vfpr_tag_out.rd;
    fpu_tag_in.fpr_windex = vfpr_tag_out.fpr_windex;
    fpu_tag_in.acc = vfpr_tag_out.rd_is_acc;

    lsu_tag_in.rd = vfpr_tag_out.rd;
    lsu_tag_in.fpr_windex = vfpr_tag_out.fpr_windex;

    vfpr_tag_in.is_store = 1'b0;
    vfpr_tag_in.is_load = 1'b0;
    vfpr_tag_in.ls_size = Word;
    vfpr_tag_in.rd_is_acc = 1'b0; // RD is on accelerator bus

    // Destination register is in FPR
    vfpr_tag_in.rd_is_fp = 1'b1;
    csr_instr = 1'b0; // is a csr instruction
    // SSR register
    ssr_active_d = ssr_active_q;
    unique casez (acc_req_q.data_op)
      // FP - FP Operations
      // Single Precision
      riscv_instr::FADD_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
      end
      riscv_instr::FSUB_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::FMUL_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MUL;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
      end
      riscv_instr::FDIV_S: begin  // currently illegal
        vfpr_tag_in.fpu_op = fpnew_pkg::DIV;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
      end
      riscv_instr::FSGNJ_S,
      riscv_instr::FSGNJN_S,
      riscv_instr::FSGNJX_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
      end
      riscv_instr::FMIN_S,
      riscv_instr::FMAX_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MINMAX;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
      end
      riscv_instr::FSQRT_S: begin  // currently illegal
        vfpr_tag_in.fpu_op = fpnew_pkg::SQRT;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegA;
      end
      riscv_instr::FMADD_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
      end
      riscv_instr::FMSUB_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.op_mode      = 1'b1;
      end
      riscv_instr::FNMSUB_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
      end
      riscv_instr::FNMADD_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.op_mode      = 1'b1;
      end
      // Vectorial Single Precision
      riscv_instr::VFADD_S,
      riscv_instr::VFADD_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFADD_R_S}) vfpr_tag_in.op_select[2] = RegBRep;
      end
      riscv_instr::VFSUB_S,
      riscv_instr::VFSUB_R_S: begin
        vfpr_tag_in.fpu_op  = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSUB_R_S}) vfpr_tag_in.op_select[2] = RegBRep;
      end
      riscv_instr::VFMUL_S,
      riscv_instr::VFMUL_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MUL;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMUL_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFDIV_S,
      riscv_instr::VFDIV_R_S: begin  // currently illegal
        vfpr_tag_in.fpu_op = fpnew_pkg::DIV;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFDIV_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFMIN_S,
      riscv_instr::VFMIN_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MINMAX;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RNE;
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMIN_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFMAX_S,
      riscv_instr::VFMAX_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MINMAX;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RTZ;
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMAX_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSQRT_S: begin // currently illegal
        vfpr_tag_in.fpu_op = fpnew_pkg::SQRT;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFMAC_S,
      riscv_instr::VFMAC_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMAC_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFMRE_S,
      riscv_instr::VFMRE_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMRE_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJ_S,
      riscv_instr::VFSGNJ_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RNE;
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSGNJ_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJN_S,
      riscv_instr::VFSGNJN_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RTZ;
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSGNJN_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJX_S,
      riscv_instr::VFSGNJX_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RDN;
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSGNJX_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSUM_S,
      riscv_instr::VFNSUM_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::VSUM;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFNSUM_S}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::VFCPKA_S_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CPKAB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFCPKA_S_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CPKAB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
      end
      // Double Precision
      riscv_instr::FADD_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FSUB_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.op_mode = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FMUL_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MUL;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FDIV_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::DIV;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FSGNJ_D,
      riscv_instr::FSGNJN_D,
      riscv_instr::FSGNJX_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FMIN_D,
      riscv_instr::FMAX_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MINMAX;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FSQRT_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SQRT;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FMADD_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FMSUB_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FNMSUB_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FNMADD_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FCVT_S_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_D_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      // [Alternate] Half Precision
      riscv_instr::FADD_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FSUB_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.op_mode = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FMUL_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MUL;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FDIV_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::DIV;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FSGNJ_H,
      riscv_instr::FSGNJN_H,
      riscv_instr::FSGNJX_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FMIN_H,
      riscv_instr::FMAX_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MINMAX;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FSQRT_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SQRT;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FMADD_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FMSUB_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FNMSUB_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FNMADD_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::VFSUM_H,
      riscv_instr::VFNSUM_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::VSUM;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFNSUM_H}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::FMULEX_S_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MUL;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FMACEX_S_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_S_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_H_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
      end
      riscv_instr::FCVT_D_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FCVT_H_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
      end
      riscv_instr::FCVT_H_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
      end
      // Vectorial [alternate] Half Precision
      riscv_instr::VFADD_H,
      riscv_instr::VFADD_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFADD_R_H}) vfpr_tag_in.op_select[2] = RegBRep;
      end
      riscv_instr::VFSUB_H,
      riscv_instr::VFSUB_R_H: begin
        vfpr_tag_in.fpu_op  = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSUB_R_H}) vfpr_tag_in.op_select[2] = RegBRep;
      end
      riscv_instr::VFMUL_H,
      riscv_instr::VFMUL_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MUL;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMUL_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFDIV_H,
      riscv_instr::VFDIV_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::DIV;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFDIV_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFMIN_H,
      riscv_instr::VFMIN_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MINMAX;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMIN_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFMAX_H,
      riscv_instr::VFMAX_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MINMAX;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RTZ;
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMAX_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSQRT_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SQRT;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFMAC_H,
      riscv_instr::VFMAC_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMAC_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFMRE_H,
      riscv_instr::VFMRE_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMRE_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJ_H,
      riscv_instr::VFSGNJ_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSGNJ_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJN_H,
      riscv_instr::VFSGNJN_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RTZ;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSGNJN_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJX_H,
      riscv_instr::VFSGNJX_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RDN;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSGNJX_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFCPKA_H_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CPKAB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFCPKB_H_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CPKAB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFCVT_S_H,
      riscv_instr::VFCVTU_S_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVTU_S_H}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_H_S,
      riscv_instr::VFCVTU_H_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVTU_H_S}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::VFCPKA_H_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CPKAB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFCPKB_H_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CPKAB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFDOTPEX_S_H,
      riscv_instr::VFDOTPEX_S_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SDOTP;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFDOTPEX_S_R_H}) vfpr_tag_in.op_select[2] = RegBRep;
      end
      riscv_instr::VFNDOTPEX_S_H,
      riscv_instr::VFNDOTPEX_S_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SDOTP;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFNDOTPEX_S_R_H}) vfpr_tag_in.op_select[2] = RegBRep;
      end
      riscv_instr::VFSUMEX_S_H,
      riscv_instr::VFNSUMEX_S_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::EXVSUM;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFNSUMEX_S_H}) vfpr_tag_in.op_mode = 1'b1;
      end
      // [Alternate] Quarter Precision
      riscv_instr::FADD_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FSUB_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.op_mode = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FMUL_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MUL;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FDIV_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::DIV;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FSGNJ_B,
      riscv_instr::FSGNJN_B,
      riscv_instr::FSGNJX_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FMIN_B,
      riscv_instr::FMAX_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MINMAX;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FSQRT_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SQRT;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FMADD_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FMSUB_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FNMSUB_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FNMADD_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegC;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::VFSUM_B,
      riscv_instr::VFNSUM_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::VSUM;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFNSUM_B}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::FMULEX_S_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MUL;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FMACEX_S_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_S_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_B_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
      end
      riscv_instr::FCVT_D_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FCVT_B_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
      end
      riscv_instr::FCVT_H_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
      end
      riscv_instr::FCVT_B_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
      end
      // Vectorial [alternate] Quarter Precision
      riscv_instr::VFADD_B,
      riscv_instr::VFADD_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFADD_R_B}) vfpr_tag_in.op_select[2] = RegBRep;
      end
      riscv_instr::VFSUB_B,
      riscv_instr::VFSUB_R_B: begin
        vfpr_tag_in.fpu_op  = fpnew_pkg::ADD;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.op_select[2] = RegB;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSUB_R_B}) vfpr_tag_in.op_select[2] = RegBRep;
      end
      riscv_instr::VFMUL_B,
      riscv_instr::VFMUL_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MUL;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMUL_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFDIV_B,
      riscv_instr::VFDIV_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::DIV;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFDIV_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFMIN_B,
      riscv_instr::VFMIN_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MINMAX;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMIN_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFMAX_B,
      riscv_instr::VFMAX_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::MINMAX;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RTZ;
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMAX_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSQRT_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SQRT;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFMAC_B,
      riscv_instr::VFMAC_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FMADD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMAC_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFMRE_B,
      riscv_instr::VFMRE_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::FNMSUB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFMRE_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJ_B,
      riscv_instr::VFSGNJ_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSGNJ_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJN_B,
      riscv_instr::VFSGNJN_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RTZ;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSGNJN_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJX_B,
      riscv_instr::VFSGNJX_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RDN;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFSGNJX_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFCPKA_B_S,
      riscv_instr::VFCPKB_B_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CPKAB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCPKB_B_S}) vfpr_tag_in.op_mode = 1;
      end
      riscv_instr::VFCPKC_B_S,
      riscv_instr::VFCPKD_B_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CPKCD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCPKD_B_S}) vfpr_tag_in.op_mode = 1;
      end
     riscv_instr::VFCPKA_B_D,
      riscv_instr::VFCPKB_B_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CPKAB;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCPKB_B_D}) vfpr_tag_in.op_mode = 1;
      end
      riscv_instr::VFCPKC_B_D,
      riscv_instr::VFCPKD_B_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CPKCD;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCPKD_B_D}) vfpr_tag_in.op_mode = 1;
      end
      riscv_instr::VFCVT_S_B,
      riscv_instr::VFCVTU_S_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVTU_S_B}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_B_S,
      riscv_instr::VFCVTU_B_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVTU_B_S}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_H_H,
      riscv_instr::VFCVTU_H_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVTU_H_H}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_H_B,
      riscv_instr::VFCVTU_H_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVTU_H_B}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_B_H,
      riscv_instr::VFCVTU_B_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVTU_B_H}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_B_B,
      riscv_instr::VFCVTU_B_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2F;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVTU_B_B}) vfpr_tag_in.op_mode = 1'b1;
      end
      riscv_instr::VFDOTPEX_H_B,
      riscv_instr::VFDOTPEX_H_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SDOTP;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFDOTPEX_H_R_B}) vfpr_tag_in.op_select[2] = RegBRep;
      end
      riscv_instr::VFNDOTPEX_H_B,
      riscv_instr::VFNDOTPEX_H_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SDOTP;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.op_mode      = 1'b1;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFNDOTPEX_H_R_B}) vfpr_tag_in.op_select[2] = RegBRep;
      end
      riscv_instr::VFSUMEX_H_B,
      riscv_instr::VFNSUMEX_H_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::EXVSUM;
        vfpr_tag_in.op_select[0] = RegA;
        vfpr_tag_in.op_select[2] = RegDest;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFNSUMEX_H_B}) vfpr_tag_in.op_mode = 1'b1;
      end
      // -------------------
      // From float to int
      // -------------------
      // Single Precision Floating-Point
      riscv_instr::FLE_S,
      riscv_instr::FLT_S,
      riscv_instr::FEQ_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::FCLASS_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CLASSIFY;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::FCVT_W_S,
      riscv_instr::FCVT_WU_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2I;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::FCVT_WU_S}) vfpr_tag_in.op_mode = 1'b1; // unsigned
      end
      riscv_instr::FMV_X_W: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.op_mode        = 1'b1; // sign-extend result
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      // Vectorial Single Precision
      riscv_instr::VFEQ_S,
      riscv_instr::VFEQ_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RDN;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFEQ_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFNE_S,
      riscv_instr::VFNE_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RDN;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.op_mode        = 1'b1;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFNE_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFLT_S,
      riscv_instr::VFLT_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RTZ;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFLT_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFGE_S,
      riscv_instr::VFGE_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RTZ;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.op_mode        = 1'b1;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFGE_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFLE_S,
      riscv_instr::VFLE_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFLE_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFGT_S,
      riscv_instr::VFGT_R_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.op_mode        = 1'b1;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFGT_R_S}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFCLASS_S: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CLASSIFY;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP32;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      // Double Precision Floating-Point
      riscv_instr::FLE_D,
      riscv_instr::FLT_D,
      riscv_instr::FEQ_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP64;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::FCLASS_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CLASSIFY;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP64;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::FCVT_W_D,
      riscv_instr::FCVT_WU_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2I;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP64;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::FCVT_WU_D}) vfpr_tag_in.op_mode = 1'b1; // unsigned
      end
      riscv_instr::FMV_X_D: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP64;
        vfpr_tag_in.op_mode        = 1'b1; // sign-extend result
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      // [Alternate] Half Precision Floating-Point
      riscv_instr::FLE_H,
      riscv_instr::FLT_H,
      riscv_instr::FEQ_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::FCLASS_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CLASSIFY;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::FCVT_W_H,
      riscv_instr::FCVT_WU_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2I;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::FCVT_WU_H}) vfpr_tag_in.op_mode = 1'b1; // unsigned
      end
      riscv_instr::FMV_X_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.op_mode        = 1'b1; // sign-extend result
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      // Vectorial [alternate] Half Precision
      riscv_instr::VFEQ_H,
      riscv_instr::VFEQ_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RDN;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFEQ_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFNE_H,
      riscv_instr::VFNE_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RDN;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.op_mode        = 1'b1;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFNE_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFLT_H,
      riscv_instr::VFLT_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RTZ;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFLT_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFGE_H,
      riscv_instr::VFGE_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RTZ;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.op_mode        = 1'b1;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFGE_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFLE_H,
      riscv_instr::VFLE_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFLE_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFGT_H,
      riscv_instr::VFGT_R_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.op_mode        = 1'b1;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFGT_R_H}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFCLASS_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CLASSIFY;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::VFMV_X_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.op_mode        = 1'b1; // sign-extend result
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::VFCVT_X_H,
      riscv_instr::VFCVT_XU_H: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2I;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.int_fmt        = fpnew_pkg::INT16;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        vfpr_tag_in.set_dyn_rm     = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVT_XU_H}) vfpr_tag_in.op_mode = 1'b1; // upper
      end
      // [Alternate] Quarter Precision Floating-Point
      riscv_instr::FLE_B,
      riscv_instr::FLT_B,
      riscv_instr::FEQ_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::FCLASS_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CLASSIFY;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::FCVT_W_B,
      riscv_instr::FCVT_WU_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2I;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::FCVT_WU_B}) vfpr_tag_in.op_mode = 1'b1; // unsigned
      end
      riscv_instr::FMV_X_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.op_mode        = 1'b1; // sign-extend result
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      // Vectorial Quarter Precision
      riscv_instr::VFEQ_B,
      riscv_instr::VFEQ_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RDN;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFEQ_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFNE_B,
      riscv_instr::VFNE_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RDN;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.op_mode        = 1'b1;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFNE_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFLT_B,
      riscv_instr::VFLT_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RTZ;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFLT_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFGE_B,
      riscv_instr::VFGE_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RTZ;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.op_mode        = 1'b1;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFGE_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFLE_B,
      riscv_instr::VFLE_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFLE_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFGT_B,
      riscv_instr::VFGT_R_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CMP;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.op_select[1]   = RegB;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.op_mode        = 1'b1;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        if (acc_req_q.data_op inside {riscv_instr::VFGT_R_B}) vfpr_tag_in.op_select[1] = RegBRep;
      end
      riscv_instr::VFCLASS_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::CLASSIFY;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RNE;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::VFMV_X_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.op_mode        = 1'b1; // sign-extend result
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
      end
      riscv_instr::VFCVT_X_B,
      riscv_instr::VFCVT_XU_B: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::F2I;
        vfpr_tag_in.op_select[0]   = RegA;
        vfpr_tag_in.src_fmt        = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          vfpr_tag_in.src_fmt      = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        vfpr_tag_in.int_fmt        = fpnew_pkg::INT8;
        vfpr_tag_in.vectorial_op   = 1'b1;
        vfpr_tag_in.rd_is_acc = 1'b1;
        vfpr_tag_in.rd_is_fp       = 1'b0;
        vfpr_tag_in.set_dyn_rm     = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVT_XU_B}) vfpr_tag_in.op_mode = 1'b1; // upper
      end
      // -------------------
      // From int to float
      // -------------------
      // Single Precision Floating-Point
      riscv_instr::FMV_W_X: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP32;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_S_W,
      riscv_instr::FCVT_S_WU: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::I2F;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP32;
        if (acc_req_q.data_op inside {riscv_instr::FCVT_S_WU}) vfpr_tag_in.op_mode = 1'b1; // unsigned
      end
      // Double Precision Floating-Point
      riscv_instr::FCVT_D_W,
      riscv_instr::FCVT_D_WU: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::I2F;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP64;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP64;
        if (acc_req_q.data_op inside {riscv_instr::FCVT_D_WU}) vfpr_tag_in.op_mode = 1'b1; // unsigned
      end
      // [Alternate] Half Precision Floating-Point
      riscv_instr::FMV_H_X: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FCVT_H_W,
      riscv_instr::FCVT_H_WU: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::I2F;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        if (acc_req_q.data_op inside {riscv_instr::FCVT_H_WU}) vfpr_tag_in.op_mode = 1'b1; // unsigned
      end
      // Vectorial Half Precision Floating-Point
      riscv_instr::VFMV_H_X: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.vectorial_op = 1'b1;
      end
      riscv_instr::VFCVT_H_X,
      riscv_instr::VFCVT_H_XU: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::I2F;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP16;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP16ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        vfpr_tag_in.int_fmt      = fpnew_pkg::INT16;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVT_H_XU}) vfpr_tag_in.op_mode = 1'b1; // upper
      end
      // [Alternate] Quarter Precision Floating-Point
      riscv_instr::FMV_B_X: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          vfpr_tag_in.src_fmt    = fpnew_pkg::FP8ALT;
          vfpr_tag_in.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FCVT_B_W,
      riscv_instr::FCVT_B_WU: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::I2F;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        if (acc_req_q.data_op inside {riscv_instr::FCVT_B_WU}) vfpr_tag_in.op_mode = 1'b1; // unsigned
      end
      // Vectorial Quarter Precision Floating-Point
      riscv_instr::VFMV_B_X: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::SGNJ;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.fpu_rnd_mode = fpnew_pkg::RUP; // passthrough without checking nan-box
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.vectorial_op = 1'b1;
      end
      riscv_instr::VFCVT_B_X,
      riscv_instr::VFCVT_B_XU: begin
        vfpr_tag_in.fpu_op = fpnew_pkg::I2F;
        vfpr_tag_in.op_select[0] = AccBus;
        vfpr_tag_in.src_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.dst_fmt      = fpnew_pkg::FP8;
        vfpr_tag_in.int_fmt      = fpnew_pkg::INT8;
        vfpr_tag_in.vectorial_op = 1'b1;
        vfpr_tag_in.set_dyn_rm   = 1'b1;
        if (acc_req_q.data_op inside {riscv_instr::VFCVT_B_XU}) vfpr_tag_in.op_mode = 1'b1; // upper
      end
      // -------------
      // Load / Store
      // -------------
      // Single Precision Floating-Point
      riscv_instr::FLW: begin
        vfpr_tag_in.is_load = 1'b1;
        vfpr_tag_in.is_fpu = 1'b0;
      end
      riscv_instr::FSW: begin
        vfpr_tag_in.is_store = 1'b1;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.is_fpu = 1'b0;
        vfpr_tag_in.rd_is_fp = 1'b0;
      end
      // Double Precision Floating-Point
      riscv_instr::FLD: begin
        vfpr_tag_in.is_load = 1'b1;
        vfpr_tag_in.ls_size = DoubleWord;
        vfpr_tag_in.is_fpu = 1'b0;
      end
      riscv_instr::FSD: begin
        vfpr_tag_in.is_store = 1'b1;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.ls_size = DoubleWord;
        vfpr_tag_in.is_fpu = 1'b0;
        vfpr_tag_in.rd_is_fp = 1'b0;
      end
      // [Alternate] Half Precision Floating-Point
      riscv_instr::FLH: begin
        vfpr_tag_in.is_load = 1'b1;
        vfpr_tag_in.ls_size = HalfWord;
        vfpr_tag_in.is_fpu = 1'b0;
      end
      riscv_instr::FSH: begin
        vfpr_tag_in.is_store = 1'b1;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.ls_size = HalfWord;
        vfpr_tag_in.is_fpu = 1'b0;
        vfpr_tag_in.rd_is_fp = 1'b0;
      end
      // [Alternate] Quarter Precision Floating-Point
      riscv_instr::FLB: begin
        vfpr_tag_in.is_load = 1'b1;
        vfpr_tag_in.ls_size = Byte;
        vfpr_tag_in.is_fpu = 1'b0;
      end
      riscv_instr::FSB: begin
        vfpr_tag_in.is_store = 1'b1;
        vfpr_tag_in.op_select[1] = RegB;
        vfpr_tag_in.ls_size = Byte;
        vfpr_tag_in.is_fpu = 1'b0;
        vfpr_tag_in.rd_is_fp = 1'b0;
      end
      // -------------
      // CSR Handling
      // -------------
      // Set or clear corresponding CSR
      riscv_instr::CSRRSI: begin
        vfpr_tag_in.is_fpu = 1'b0;
        vfpr_tag_in.rd_is_fp = 1'b0;
        csr_instr = 1'b1;
        ssr_active_d |= rs1[0];
      end
      riscv_instr::CSRRCI: begin
        vfpr_tag_in.is_fpu = 1'b0;
        vfpr_tag_in.rd_is_fp = 1'b0;
        csr_instr = 1'b1;
        ssr_active_d &= ~rs1[0];
      end
      default: begin
        vfpr_tag_in.is_fpu = 1'b0;
        acc_resp_o.error = 1'b1;
        vfpr_tag_in.rd_is_fp = 1'b0;
      end
    endcase
    // fix round mode for vectors and fp16alt
    if (vfpr_tag_in.set_dyn_rm) vfpr_tag_in.fpu_rnd_mode = fpu_rnd_mode_i;
    // check if vfpr_tag_in.src_fmt or vfpr_tag_in.dst_fmt is acutually the alternate version
    // single-format float operations ignore fpu_fmt_mode_i.src
    // reason: for performance reasons when mixing expanding and non-expanding operations
    if (vfpr_tag_in.src_fmt == fpnew_pkg::FP16 && fpu_fmt_mode_i.src == 1'b1) vfpr_tag_in.src_fmt = fpnew_pkg::FP16ALT;
    if (vfpr_tag_in.dst_fmt == fpnew_pkg::FP16 && fpu_fmt_mode_i.dst == 1'b1) vfpr_tag_in.dst_fmt = fpnew_pkg::FP16ALT;
    if (vfpr_tag_in.src_fmt == fpnew_pkg::FP8 && fpu_fmt_mode_i.src == 1'b1) vfpr_tag_in.src_fmt = fpnew_pkg::FP8ALT;
    if (vfpr_tag_in.dst_fmt == fpnew_pkg::FP8 && fpu_fmt_mode_i.dst == 1'b1) vfpr_tag_in.dst_fmt = fpnew_pkg::FP8ALT;
  end

  for (genvar i = 0; i < 3; i++) begin: gen_vfpr_op_ready
    always_comb begin
      unique case (vfpr_tag_in.op_select[i])
        None: begin
          vfpr_op_ready[i] = 1'b1;
        end
        AccBus: begin
          vfpr_op_ready[i] = acc_req_valid_q;
        end
        RegA, RegB, RegBRep, RegC, RegDest: begin
          vfpr_op_ready[i] = ~sb_collision[i];
        end
        default: begin
          vfpr_op_ready[i] = 1'b1;
        end
      endcase
    end
  end

  // snitch_vfpr #(
  //   .DataWidth(DataWidth),
  //   .TCDMMemAddrWidth(TCDMMemAddrWidth),
  //   .tcdm_req_t(tcdm_req_t),
  //   .tcdm_rsp_t(tcdm_rsp_t),
  //   .mem_req_t(mem_req_t),
  //   .mem_rsp_t(mem_rsp_t),
  //   .tcdm_user_t(tcdm_user_t)
  // ) i_vfpr (
  //   .clk_i,
  //   .rst_i,
  //   .wr_req_i(data_req_i),
  //   .wr_rsp_o(data_rsp_o),
  //   .mem_req_o(mem_req_o),
  //   .mem_rsp_i(mem_rsp_i)
  // );

  logic [2:0][4:0] vfpr_rdata;
  regaddr_t [2:0] reg_addrs;
  assign reg_addrs[0] = vfpr_tag_in.fpr_raddr[0];
  assign reg_addrs[1] = vfpr_tag_in.fpr_raddr[1];
  assign reg_addrs[2] = vfpr_tag_in.fpr_raddr[2];

  snitch_vfpr #(
    .DataWidth(DataWidth),
    .TCDMMemAddrWidth(TCDMMemAddrWidth),
    .tcdm_req_t(tcdm_req_t),
    .tcdm_rsp_t(tcdm_rsp_t),
    .mem_req_t(mem_req_t),
    .mem_rsp_t(mem_rsp_t),
    .tcdm_user_t(tcdm_user_t),
    .tag_t(vfpr_tag_t)
  ) i_vfpr (
    .clk_i,
    .rst_i,
    .raddr_i(reg_addrs),
    .ren_i({
      vfpr_tag_in.op_select[2] inside {RegA, RegB, RegBRep, RegC, RegDest},
      vfpr_tag_in.op_select[1] inside {RegA, RegB, RegBRep, RegC, RegDest},
      vfpr_tag_in.op_select[0] inside {RegA, RegB, RegBRep, RegC, RegDest}
    }),
    .rvalid_i(vfpr_in_valid),
    .rready_o(vfpr_in_ready),
    .rdata_o(vfpr_rdata),
    .rvalid_o(vfpr_out_valid),
    .rready_i(vfpr_out_ready),
    .rtag_i(vfpr_tag_in),
    .rtag_o(vfpr_tag_out),
    .wr_req_i(data_req_i),
    .wr_rsp_o(data_rsp_o),
    .mem_req_o(mem_req_o),
    .mem_rsp_i(mem_rsp_i),
    .vfpr_tracer_port(vfpr_tracer_port_o)
  );

  snitch_regfile #(
    .DATA_WIDTH     ( FLEN ),
    .NR_READ_PORTS  ( 3    ),
    .NR_WRITE_PORTS ( 1    ),
    .ZERO_REG_ZERO  ( 0    ),
    .ADDR_WIDTH     ( 5    )
  ) i_ff_regfile (
    .clk_i,
    .raddr_i   ( vfpr_tag_out.fpr_raddr ),
    .rdata_o   ( fpr_rdata ),
    .waddr_i   ( fpr_waddr ),
    .wdata_i   ( fpr_wdata ),
    .we_i      ( fpr_we    )
  );

  // ----------------------
  // Operand Select
  // ----------------------
  logic [2:0][FLEN-1:0] acc_qdata;
  assign acc_qdata = {vfpr_tag_out.data_argc, vfpr_tag_out.data_argb, vfpr_tag_out.data_arga};

  // Mux address lines as operands for the FPU can be mangled
  always_comb begin
    vfpr_tag_in.fpr_raddr[0] = rs1;
    vfpr_tag_in.fpr_raddr[1] = rs2;
    vfpr_tag_in.fpr_raddr[2] = rs3;

    unique case (vfpr_tag_in.op_select[1])
      RegA: begin
        vfpr_tag_in.fpr_raddr[1] = rs1;
      end
      default:;
    endcase

    unique case (vfpr_tag_in.op_select[2])
      RegB,
      RegBRep: begin
        vfpr_tag_in.fpr_raddr[2] = rs2;
      end
      RegDest: begin
        vfpr_tag_in.fpr_raddr[2] = vfpr_tag_in.rd;
      end
      default:;
    endcase
  end

  for (genvar i = 0; i < 3; i++) begin: gen_operand_select
    always_comb begin
      unique case (vfpr_tag_out.op_select[i])
        None: begin
          op[i] = '1;
        end
        AccBus: begin
          op[i] = acc_qdata[i];
        end
        // Scoreboard or SSR
        RegA, RegB, RegBRep, RegC, RegDest: begin
          // map register 0 and 1 to SSRs
          op[i] = fpr_rdata[i];
          // Replicate if needed
          if (vfpr_tag_out.op_select[i] == RegBRep) begin
            unique case (vfpr_tag_out.src_fmt)
              fpnew_pkg::FP32:    op[i] = {(FLEN / 32){op[i][31:0]}};
              fpnew_pkg::FP16,
              fpnew_pkg::FP16ALT: op[i] = {(FLEN / 16){op[i][15:0]}};
              fpnew_pkg::FP8,
              fpnew_pkg::FP8ALT:  op[i] = {(FLEN /  8){op[i][ 7:0]}};
              default:            op[i] = op[i][FLEN-1:0];
            endcase
          end
        end
        default: begin
          op[i] = '0;
        end
      endcase
    end
  end

  // ----------------------
  // Floating Point Unit
  // ----------------------
  snitch_fpu #(
    .tag_t   ( tag_t   ),
    .RVF     ( RVF     ),
    .RVD     ( RVD     ),
    .XF16    ( XF16    ),
    .XF16ALT ( XF16ALT ),
    .XF8     ( XF8     ),
    .XF8ALT  ( XF8ALT  ),
    .XFVEC   ( XFVEC   ),
    .FLEN    ( FLEN    ),
    .FPUImplementation  (FPUImplementation),
    .RegisterFPUIn      (RegisterFPUIn),
    .RegisterFPUOut     (RegisterFPUOut)
  ) i_fpu (
    .clk_i                           ,
    .rst_ni         ( ~rst_i        ),
    .hart_id_i      ( hart_id_i     ),
    .operands_i     ( op            ),
    .rnd_mode_i     ( vfpr_tag_out.fpu_rnd_mode  ),
    .op_i           ( vfpr_tag_out.fpu_op        ),
    .op_mod_i       ( vfpr_tag_out.op_mode       ), // Sign of operand?
    .src_fmt_i      ( vfpr_tag_out.src_fmt       ),
    .dst_fmt_i      ( vfpr_tag_out.dst_fmt       ),
    .int_fmt_i      ( vfpr_tag_out.int_fmt       ),
    .vectorial_op_i ( vfpr_tag_out.vectorial_op  ),
    .tag_i          ( fpu_tag_in    ),
    .in_valid_i     ( fpu_in_valid  ),
    .in_ready_o     ( fpu_in_ready  ),
    .result_o       ( fpu_result    ),
    .status_o       ( fpu_status_o  ),
    .tag_o          ( fpu_tag_out   ),
    .out_valid_o    ( fpu_out_valid ),
    .out_ready_i    ( fpu_out_ready )
  );

  assign ssr_waddr_o = fpr_waddr;
  assign ssr_wdata_o = fpr_wdata;
  logic [63:0] nan_boxed_arga;
  // this datapath bypasses vfpr
  assign nan_boxed_arga = {{32{1'b1}}, acc_req_q.data_arga[31:0]};

  // Arbitrate Register File Write Port
  always_comb begin
    fpr_we = 1'b0;
    fpr_waddr = '0;
    fpr_wdata = '0;
    fpr_wvalid = 1'b0;
    lsu_pready = 1'b0;
    fpr_wready = 1'b1;
    ssr_wvalid_o = 1'b0;
    ssr_wdone_o = 1'b1;
    sb_pop_valid = 1'b0;
    acc_dp_ready = '0;
    fpr_wr_req.q_valid = 1'b0;
    // the accelerator master wants to write
    if (acc_dp_valid) begin
      acc_dp_ready = '1;
      fpr_we = 1'b1;
      // NaN-Box the value
      fpr_wdata = nan_boxed_arga[FLEN-1:0];
      fpr_wr_req.q.data = nan_boxed_arga[FLEN-1:0];
      fpr_waddr = vfpr_tag_in.rd;
      fpr_wr_req.q.addr = vfpr_tag_in.rd;
      fpr_wvalid = 1'b1;
      fpr_wr_req.q_valid = 1'b1;
      fpr_wready = 1'b0;
    end else if (fpu_out_valid && !fpu_tag_out.acc) begin
      fpr_we = 1'b1;
      if (fpu_tag_out.ssr) begin
        ssr_wvalid_o = 1'b1;
        // stall write-back to SSR
        if (!ssr_wready_i) begin
          fpr_wready = 1'b0;
          fpr_we = 1'b0;
        end else begin
          ssr_wdone_o = 1'b1;
        end
      end
      fpr_wdata = fpu_result;
      fpr_waddr = fpu_tag_out.rd;
      fpr_windex = fpu_tag_out.fpr_windex;
      fpr_wvalid = 1'b1;
      sb_pop_valid = fpu_out_ready;
    end else if (lsu_pvalid) begin
      lsu_pready = 1'b1;
      fpr_we = 1'b1;
      fpr_wdata = ld_result;
      fpr_waddr = lsu_tag_out.rd;
      fpr_windex = lsu_tag_out.fpr_windex;
      fpr_wvalid = 1'b1;
      fpr_wready = 1'b0;
      sb_pop_valid = lsu_pready;
    end
  end

  // ----------------------
  // Load/Store Unit
  // ----------------------

  snitch_lsu #(
    .AddrWidth (AddrWidth),
    .DataWidth (DataWidth),
    .dreq_t (dreq_t),
    .drsp_t (drsp_t),
    .tag_t (tag_t),
    .NumOutstandingMem (NumFPOutstandingMem),
    .NumOutstandingLoads (NumFPOutstandingLoads),
    .NaNBox (1'b1)
  ) i_snitch_lsu (
    .clk_i (clk_i),
    .rst_i (rst_i),
    .lsu_qtag_i (lsu_tag_in),
    .lsu_qwrite_i (vfpr_tag_out.is_store),
    .lsu_qsigned_i (1'b1), // all floating point loads are signed
    .lsu_qaddr_i (vfpr_tag_out.data_argc[AddrWidth-1:0]),
    .lsu_qdata_i (op[1]),
    .lsu_qsize_i (vfpr_tag_out.ls_size),
    .lsu_qamo_i (reqrsp_pkg::AMONone),
    .lsu_qvalid_i (lsu_in_valid),
    .lsu_qready_o (lsu_in_ready),
    .lsu_pdata_o (ld_result),
    .lsu_ptag_o (lsu_tag_out),
    .lsu_perror_o (), // ignored for the moment
    .lsu_pvalid_o (lsu_pvalid),
    .lsu_pready_i (lsu_pready),
    .lsu_empty_o (/* unused */),
    .data_req_o,
    .data_rsp_i
  );

  // SSRs
  for (genvar i = 0; i < 3; i++) begin 
    assign ssr_rdone_o[i] = '0;
    assign ssr_rvalid_o[i] = '0;
  end
  assign ssr_raddr_o = '0;
  assign ssr_wdata_o = '0;
  assign ssr_wvalid_o = 1'b0;
  assign ssr_wdone_o = 1'b1;

  // Counter pipeline.
  logic issue_fpu, issue_core_to_fpu, issue_fpu_seq;
  `FFAR(issue_fpu, fpu_in_valid & fpu_in_ready, 1'b0, clk_i, rst_i)
  `FFAR(issue_core_to_fpu, acc_req_valid_i & acc_req_ready_o, 1'b0, clk_i, rst_i)
  `FFAR(issue_fpu_seq, acc_req_valid & acc_req_ready, 1'b0, clk_i, rst_i)

  always_comb begin
    core_events_o = '0;
    core_events_o.issue_fpu = issue_fpu;
    core_events_o.issue_core_to_fpu = issue_core_to_fpu;
    core_events_o.issue_fpu_seq = issue_fpu_seq;
  end

  // Tracer
  // pragma translate_off
  assign trace_port_o.source       = snitch_pkg::SrcFpu;
  assign trace_port_o.acc_q_hs     = (acc_req_valid_q  && acc_req_ready_q );
  assign trace_port_o.fpu_out_hs   = (fpu_out_valid && fpu_out_ready );
  assign trace_port_o.lsu_q_hs     = (lsu_in_valid    && lsu_in_ready    );
  assign trace_port_o.op_in        = acc_req_q.data_op;
  assign trace_port_o.rs1          = rs1;
  assign trace_port_o.rs2          = rs2;
  assign trace_port_o.rs3          = rs3;
  assign trace_port_o.rd           = vfpr_tag_out.rd;
  assign trace_port_o.op_sel_0     = vfpr_tag_out.op_select[0];
  assign trace_port_o.op_sel_1     = vfpr_tag_out.op_select[1];
  assign trace_port_o.op_sel_2     = vfpr_tag_out.op_select[2];
  assign trace_port_o.src_fmt      = vfpr_tag_out.src_fmt;
  assign trace_port_o.dst_fmt      = vfpr_tag_out.dst_fmt;
  assign trace_port_o.int_fmt      = vfpr_tag_out.int_fmt;
  assign trace_port_o.acc_qdata_0  = acc_qdata[0];
  assign trace_port_o.acc_qdata_1  = acc_qdata[1];
  assign trace_port_o.acc_qdata_2  = acc_qdata[2];
  assign trace_port_o.op_0         = op[0];
  assign trace_port_o.op_1         = op[1];
  assign trace_port_o.op_2         = op[2];
  assign trace_port_o.use_fpu      = vfpr_tag_out.is_fpu;
  assign trace_port_o.fpu_in_rd    = fpu_tag_in.rd;
  assign trace_port_o.fpu_in_acc   = vfpr_tag_in.rd_is_acc;
  assign trace_port_o.ls_size      = vfpr_tag_out.ls_size;
  assign trace_port_o.is_load      = vfpr_tag_out.is_load;
  assign trace_port_o.is_store     = vfpr_tag_out.is_store;
  assign trace_port_o.lsu_qaddr    = i_snitch_lsu.lsu_qaddr_i;
  assign trace_port_o.lsu_rd       = lsu_tag_out.rd;
  assign trace_port_o.acc_wb_ready = (result_select == ResAccBus);
  assign trace_port_o.fpu_out_acc  = fpu_tag_out.acc;
  assign trace_port_o.fpu_out_rd   = fpu_tag_out.rd;
  assign trace_port_o.fpr_waddr    = fpr_waddr;
  assign trace_port_o.fpr_wdata    = fpr_wdata[0];
  assign trace_port_o.fpr_we       = fpr_we[0];
  // pragma translate_on

  /// Assertions
  `ASSERT(RegWriteKnown, fpr_we |-> !$isunknown(fpr_wdata), clk_i, rst_i)
endmodule
