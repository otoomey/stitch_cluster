// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// [x] Updated stitch-sequencer to support adding fpu id to int pipe. regs
// [x] Offset register configuration
// [ ] Add register loading mechanism
// [ ] Update scoreboard
// [ ] Add register write mechanism


`include "common_cells/registers.svh"
`include "common_cells/assertions.svh"

// Floating Point Subsystem
module stitch_fp_ss import snitch_pkg::*; #(
  parameter int unsigned AddrWidth = 32,
  parameter int unsigned BankAddrWidth = 0,
  parameter int unsigned DataWidth = 32,
  parameter int unsigned NumFPOutstandingLoads = 0,
  parameter int unsigned NumFPOutstandingMem = 0,
  parameter int unsigned NumFPUSequencerInstr = 0,
  parameter int unsigned ScoreboardDepth = 0,
  parameter int unsigned DefaultRegOffset = 2**BankAddrWidth-32,
  parameter type dreq_t = logic,
  parameter type drsp_t = logic,
  parameter type dbankreq_t = logic,
  parameter type dbankrsp_t = logic,
  parameter bit RegisterSequencer = 0,
  parameter bit RegisterDecoder   = 0,
  parameter bit RegisterFPUIn     = 0,
  parameter bit RegisterFPUOut    = 0,
  parameter bit Xfrep = 1,
  parameter fpnew_pkg::fpu_implementation_t FPUImplementation = '0,
  parameter bit Xssr = 1,
//   parameter int unsigned NumSsrs = 0,
//   parameter logic [NumSsrs-1:0][4:0]  SsrRegs = '0,
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
  parameter int unsigned BankGroupAddrWidth = 4 * BankAddrWidth,
  parameter int unsigned RegsPerBanks = 32 / 4,
  parameter int unsigned RegGroupsPerBank = (2 ** BankAddrWidth) / 8
) (
  input  logic             clk_i,
  input  logic             rst_i,
  input  logic [31:0]      fpu_id_i,
  // pragma translate_off
  output fpu_trace_port_t  trace_port_o,
  output fpu_sequencer_trace_port_t sequencer_tracer_port_o,
  // pragma translate_on
  input  logic [31:0]      hart_id_i,
  // Accelerator Interface - Slave
  input  acc_req_t         acc_req_i,
  input  logic             acc_req_valid_i,
  output logic             acc_req_ready_o,
  output acc_resp_t        acc_resp_o,
  output logic             acc_resp_valid_o,
  input  logic             acc_resp_ready_i,
  // Config interface
  input  logic [4:0]               cfg_word_i,
  output logic [DataWidth-1:0]     cfg_rdata_o,
  input  logic [DataWidth-1:0]     cfg_wdata_i,
  input  logic                     cfg_write_i,
  output logic                     cfg_wready_o,
  // TCDM Data Interface for regular FP load/stores.
  output dreq_t            data_slow_req_o,
  input  drsp_t            data_slow_rsp_i,
  // Data Interface for accelerated FP load/stores.
  output dbankreq_t [3:0]            data_fast_req_o,
  input  dbankrsp_t [3:0]            data_fast_rsp_i,
  // Register Interface
  // FPU **un-timed** Side-channel
  input  fpnew_pkg::roundmode_e fpu_rnd_mode_i,
  input  fpnew_pkg::fmt_mode_t  fpu_fmt_mode_i,
  output fpnew_pkg::status_t    fpu_status_o,
  // SSR Interface
//   output logic  [2:0][4:0] ssr_raddr_o,
//   input  data_t [2:0]      ssr_rdata_i,
//   output logic  [2:0]      ssr_rvalid_o,
//   input  logic  [2:0]      ssr_rready_i,
//   output logic  [2:0]      ssr_rdone_o,
//   output logic  [0:0][4:0] ssr_waddr_o,
//   output data_t [0:0]      ssr_wdata_o,
//   output logic  [0:0]      ssr_wvalid_o,
//   input  logic  [0:0]      ssr_wready_i,
//   output logic  [0:0]      ssr_wdone_o,
  // SSR stream control interface
//   input  logic             streamctl_done_i,
//   input  logic             streamctl_valid_i,
//   output logic             streamctl_ready_o,
  // Core event strobes
  output core_events_t core_events_o
);

  logic [2:0][4:0]      fpr_raddr;
  logic [2:0][FLEN-1:0] fpr_rdata;

  logic [0:0]           fpr_we;
  logic [0:0][4:0]      fpr_waddr;
  logic [0:0][FLEN-1:0] fpr_wdata;
  logic [0:0]           fpr_wvalid;
  logic [0:0]           fpr_wready;

  // regfile offsets
  // Bank 1: f[0,1,2,3,4,5,6,7] 00000-00111
  // Bank 2: f[8,9,10,11,12,13,14,15] 01000-01111
  // Bank 3: f[16,17,18,19,20,21,22,23] 10000-10111
  // Bank 4: f[24,25,26,27,28,29,30,31] 11000-11111
  logic [3:0][RegGroupsPerBank-1:0] cfg_offsets_d, cfg_offsets_q;
  logic [3:0][RegGroupsPerBank-1:0] cfg_strides_d, cfg_strides_q;

  `FFARN(cfg_offsets_q, cfg_offsets_d, DefaultRegOffset, clk_i, rst_ni)
  `FFARN(cfg_strides_q, cfg_strides_d, '0, clk_i, rst_ni)

  typedef logic [BankGroupAddrWidth-1:0]  vfpr_addr_t;
  typedef logic [FLEN-1:0]                data_t;
  typedef logic [FLEN/8-1:0]              strb_t;
  `TCDM_TYPEDEF_ALL(vfpr, vfpr_addr_t, data_t, strb_t, logic)


//   logic ssr_active_d, ssr_active_q, ssr_active_ena;
//   `FFLAR(ssr_active_q, Xssr & ssr_active_d, ssr_active_ena, 1'b0, clk_i, rst_i)

  typedef struct packed {
    logic                          acc; // write-back to result bus
    logic [BankGroupAddrWidth-1:0] rd_addr;              
    logic [ScoreboardDepth-1:0]    rd_sb_index;
  } tag_t;

  tag_t fpu_tag_in, fpu_tag_out;

  logic use_fpu;
  logic [2:0][FLEN-1:0] op;
  logic [2:0] op_ready; // operand is ready

  logic        lsu_qready;
  logic        lsu_qvalid;
  logic [FLEN-1:0] ld_result;
  logic [4:0]  lsu_rd;
  logic        lsu_pvalid;
  logic        lsu_pready;

  logic rd_is_fp;

  logic csr_instr;

  // FPU Controller
  logic fpu_out_valid, fpu_out_ready;
  logic fpu_in_valid, fpu_in_ready;

  typedef enum logic [2:0] {
    None,
    AccBus,
    RegA, RegB, RegC,
    RegBRep, // Replication for vectors
    RegDest
  } op_select_e;
  op_select_e [2:0] op_select;

  typedef enum logic [1:0] {
    ResNone, ResAccBus
  } result_select_e;
  result_select_e result_select;

  logic [4:0] rs1, rs2, rs3, rd;

  // LSU
  typedef enum logic [1:0] {
    Byte       = 2'b00,
    HalfWord   = 2'b01,
    Word       = 2'b10,
    DoubleWord = 2'b11
  } ls_size_e;


  logic dst_ready;

  // -------------
  // FPU Sequencer
  // -------------
  acc_req_t         seq_out, seq_out_q;
  logic             seq_out_valid, seq_out_valid_q;
  logic             seq_out_ready, seq_out_ready_q;
  if (Xfrep) begin : gen_fpu_sequencer
    stitch_sequencer #(
      .AddrWidth (AddrWidth),
      .DataWidth (DataWidth),
      .Depth     (NumFPUSequencerInstr)
    ) i_stitch_fpu_sequencer (
      .clk_i,
      .rst_i,
      .fpu_id_i,
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
      .oup_qaddr_o      ( seq_out.addr        ),
      .oup_qid_o        ( seq_out.id          ),
      .oup_qdata_op_o   ( seq_out.data_op     ),
      .oup_qdata_arga_o ( seq_out.data_arga   ),
      .oup_qdata_argb_o ( seq_out.data_argb   ),
      .oup_qdata_argc_o ( seq_out.data_argc   ),
      .oup_qvalid_o     ( seq_out_valid       ),
      .oup_qready_i     ( seq_out_ready       ),
      // .streamctl_done_i,
      // .streamctl_valid_i,
      // .streamctl_ready_o
    );
  end else begin : gen_no_fpu_sequencer
    // pragma translate_off
    assign sequencer_tracer_port_o = 0;
    // pragma translate_on
    assign acc_req_ready_o = seq_out_ready;
    assign seq_out_valid = acc_req_valid_i;
    assign seq_out = acc_req_i;
  end

  // Optional spill-register
  spill_register  #(
    .T      ( acc_req_t                           ),
    .Bypass ( !RegisterSequencer || !Xfrep )
  ) i_spill_register_acc (
    .clk_i   ,
    .rst_ni  ( ~rst_i          ),
    .valid_i ( seq_out_valid   ),
    .ready_o ( seq_out_ready   ),
    .data_i  ( seq_out         ),
    .valid_o ( seq_out_valid_q ),
    .ready_i ( seq_out_ready_q ),
    .data_o  ( seq_out_q       )
  );

  // -------------
  // FPU Decoder/Scoreboard
  // -------------
  typedef struct packed {
    logic [3:0][TotalAddrWidth-1:0] fpr_bank_raddr;
    logic [2:0]                     fpr_req_enable;
    logic [2:0][1:0]                fpr_reg_coallesce;
    fpnew_pkg::operation_e          fpu_op;
    fpnew_pkg::roundmode_e          fpu_rnd_mode;
    fpnew_pkg::fp_format_e          src_fmt, dst_fmt;
    fpnew_pkg::int_format_e         int_fmt;
    logic                           vectorial_op;
    logic                           set_dyn_rm;
    tag_t                           fpu_tag_in;
    logic                           op_mode;
    logic                           is_load;
    logic                           is_store;
    ls_size_e                       ls_size;
  } sb_dec_out_t;

  logic sb_out_valid, sb_out_valid_q;
  logic sb_out_ready, sb_out_ready_q;
  sb_dec_out_t sb_out, sb_out_q;

  logic [ScoreboardDepth-1:0] rd_sb_index;
  logic [2:0] sb_hit;

  assign rd = seq_out_q.data_op[11:7];
  assign rs1 = seq_out_q.data_op[19:15];
  assign rs2 = seq_out_q.data_op[24:20];
  assign rs3 = seq_out_q.data_op[31:27];

  always_comb begin
    acc_resp_o.error = 1'b0;
    sb_out.fpu_op = fpnew_pkg::ADD;
    use_fpu = 1'b1;
    sb_out.fpu_rnd_mode = (fpnew_pkg::roundmode_e'(seq_out_q.data_op[14:12]) == fpnew_pkg::DYN)
                   ? fpu_rnd_mode_i
                   : fpnew_pkg::roundmode_e'(seq_out_q.data_op[14:12]);

    sb_out.set_dyn_rm = 1'b0;

    sb_out.src_fmt = fpnew_pkg::FP32;
    sb_out.dst_fmt = fpnew_pkg::FP32;
    sb_out.int_fmt = fpnew_pkg::INT32;

    result_select = ResNone;

    op_select[0] = None;
    op_select[1] = None;
    op_select[2] = None;

    sb_out.vectorial_op = 1'b0;
    sb_out.op_mode = 1'b0;

    sb_out.fpu_tag_in.rd_addr = fpr_bank_raddr[3];
    sb_out.fpu_tag_in.rd_sb_index = rd_sb_index;
    sb_out.fpu_tag_in.acc = 1'b0; // RD is on accelerator bus

    sb_out.is_store = 1'b0;
    sb_out.is_load = 1'b0;
    sb_out.ls_size = Word;

    // Destination register is in FPR
    rd_is_fp = 1'b1;
    csr_instr = 1'b0; // is a csr instruction
    // SSR register
    // ssr_active_d = ssr_active_q;
    unique casez (seq_out_q.data_op)
      // FP - FP Operations
      // Single Precision
      riscv_instr::FADD_S: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
      end
      riscv_instr::FSUB_S: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.op_mode = 1'b1;
      end
      riscv_instr::FMUL_S: begin
        sb_out.fpu_op = fpnew_pkg::MUL;
        op_select[0] = RegA;
        op_select[1] = RegB;
      end
      riscv_instr::FDIV_S: begin  // currently illegal
        sb_out.fpu_op = fpnew_pkg::DIV;
        op_select[0] = RegA;
        op_select[1] = RegB;
      end
      riscv_instr::FSGNJ_S,
      riscv_instr::FSGNJN_S,
      riscv_instr::FSGNJX_S: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
      end
      riscv_instr::FMIN_S,
      riscv_instr::FMAX_S: begin
        sb_out.fpu_op = fpnew_pkg::MINMAX;
        op_select[0] = RegA;
        op_select[1] = RegB;
      end
      riscv_instr::FSQRT_S: begin  // currently illegal
        sb_out.fpu_op = fpnew_pkg::SQRT;
        op_select[0] = RegA;
        op_select[1] = RegA;
      end
      riscv_instr::FMADD_S: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
      end
      riscv_instr::FMSUB_S: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.op_mode      = 1'b1;
      end
      riscv_instr::FNMSUB_S: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
      end
      riscv_instr::FNMADD_S: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.op_mode      = 1'b1;
      end
      // Vectorial Single Precision
      riscv_instr::VFADD_S,
      riscv_instr::VFADD_R_S: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFADD_R_S}) op_select[2] = RegBRep;
      end
      riscv_instr::VFSUB_S,
      riscv_instr::VFSUB_R_S: begin
        sb_out.fpu_op  = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.op_mode      = 1'b1;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSUB_R_S}) op_select[2] = RegBRep;
      end
      riscv_instr::VFMUL_S,
      riscv_instr::VFMUL_R_S: begin
        sb_out.fpu_op = fpnew_pkg::MUL;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMUL_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFDIV_S,
      riscv_instr::VFDIV_R_S: begin  // currently illegal
        sb_out.fpu_op = fpnew_pkg::DIV;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFDIV_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFMIN_S,
      riscv_instr::VFMIN_R_S: begin
        sb_out.fpu_op = fpnew_pkg::MINMAX;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RNE;
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMIN_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFMAX_S,
      riscv_instr::VFMAX_R_S: begin
        sb_out.fpu_op = fpnew_pkg::MINMAX;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RTZ;
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMAX_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSQRT_S: begin // currently illegal
        sb_out.fpu_op = fpnew_pkg::SQRT;
        op_select[0] = RegA;
        op_select[1] = RegA;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFMAC_S,
      riscv_instr::VFMAC_R_S: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMAC_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFMRE_S,
      riscv_instr::VFMRE_R_S: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMRE_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJ_S,
      riscv_instr::VFSGNJ_R_S: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RNE;
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSGNJ_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJN_S,
      riscv_instr::VFSGNJN_R_S: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RTZ;
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSGNJN_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJX_S,
      riscv_instr::VFSGNJX_R_S: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RDN;
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSGNJX_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSUM_S,
      riscv_instr::VFNSUM_S: begin
        sb_out.fpu_op = fpnew_pkg::VSUM;
        op_select[0] = RegA;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFNSUM_S}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::VFCPKA_S_S: begin
        sb_out.fpu_op = fpnew_pkg::CPKAB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFCPKA_S_D: begin
        sb_out.fpu_op = fpnew_pkg::CPKAB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
      end
      // Double Precision
      riscv_instr::FADD_D: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FSUB_D: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.op_mode = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FMUL_D: begin
        sb_out.fpu_op = fpnew_pkg::MUL;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FDIV_D: begin
        sb_out.fpu_op = fpnew_pkg::DIV;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FSGNJ_D,
      riscv_instr::FSGNJN_D,
      riscv_instr::FSGNJX_D: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FMIN_D,
      riscv_instr::FMAX_D: begin
        sb_out.fpu_op = fpnew_pkg::MINMAX;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FSQRT_D: begin
        sb_out.fpu_op = fpnew_pkg::SQRT;
        op_select[0] = RegA;
        op_select[1] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FMADD_D: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FMSUB_D: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FNMSUB_D: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FNMADD_D: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FCVT_S_D: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_D_S: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      // [Alternate] Half Precision
      riscv_instr::FADD_H: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FSUB_H: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.op_mode = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FMUL_H: begin
        sb_out.fpu_op = fpnew_pkg::MUL;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FDIV_H: begin
        sb_out.fpu_op = fpnew_pkg::DIV;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FSGNJ_H,
      riscv_instr::FSGNJN_H,
      riscv_instr::FSGNJX_H: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FMIN_H,
      riscv_instr::FMAX_H: begin
        sb_out.fpu_op = fpnew_pkg::MINMAX;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FSQRT_H: begin
        sb_out.fpu_op = fpnew_pkg::SQRT;
        op_select[0] = RegA;
        op_select[1] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FMADD_H: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FMSUB_H: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FNMSUB_H: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FNMADD_H: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::VFSUM_H,
      riscv_instr::VFNSUM_H: begin
        sb_out.fpu_op = fpnew_pkg::VSUM;
        op_select[0] = RegA;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFNSUM_H}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::FMULEX_S_H: begin
        sb_out.fpu_op = fpnew_pkg::MUL;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FMACEX_S_H: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_S_H: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_H_S: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
      end
      riscv_instr::FCVT_D_H: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FCVT_H_D: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
      end
      riscv_instr::FCVT_H_H: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
      end
      // Vectorial [alternate] Half Precision
      riscv_instr::VFADD_H,
      riscv_instr::VFADD_R_H: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFADD_R_H}) op_select[2] = RegBRep;
      end
      riscv_instr::VFSUB_H,
      riscv_instr::VFSUB_R_H: begin
        sb_out.fpu_op  = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSUB_R_H}) op_select[2] = RegBRep;
      end
      riscv_instr::VFMUL_H,
      riscv_instr::VFMUL_R_H: begin
        sb_out.fpu_op = fpnew_pkg::MUL;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMUL_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFDIV_H,
      riscv_instr::VFDIV_R_H: begin
        sb_out.fpu_op = fpnew_pkg::DIV;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFDIV_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFMIN_H,
      riscv_instr::VFMIN_R_H: begin
        sb_out.fpu_op = fpnew_pkg::MINMAX;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RNE;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMIN_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFMAX_H,
      riscv_instr::VFMAX_R_H: begin
        sb_out.fpu_op = fpnew_pkg::MINMAX;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.fpu_rnd_mode = fpnew_pkg::RTZ;
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMAX_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSQRT_H: begin
        sb_out.fpu_op = fpnew_pkg::SQRT;
        op_select[0] = RegA;
        op_select[1] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFMAC_H,
      riscv_instr::VFMAC_R_H: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMAC_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFMRE_H,
      riscv_instr::VFMRE_R_H: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMRE_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJ_H,
      riscv_instr::VFSGNJ_R_H: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RNE;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSGNJ_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJN_H,
      riscv_instr::VFSGNJN_R_H: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RTZ;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSGNJN_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJX_H,
      riscv_instr::VFSGNJX_R_H: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RDN;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSGNJX_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFCPKA_H_S: begin
        sb_out.fpu_op = fpnew_pkg::CPKAB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFCPKB_H_S: begin
        sb_out.fpu_op = fpnew_pkg::CPKAB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFCVT_S_H,
      riscv_instr::VFCVTU_S_H: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVTU_S_H}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_H_S,
      riscv_instr::VFCVTU_H_S: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVTU_H_S}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::VFCPKA_H_D: begin
        sb_out.fpu_op = fpnew_pkg::CPKAB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFCPKB_H_D: begin
        sb_out.fpu_op = fpnew_pkg::CPKAB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFDOTPEX_S_H,
      riscv_instr::VFDOTPEX_S_R_H: begin
        sb_out.fpu_op = fpnew_pkg::SDOTP;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFDOTPEX_S_R_H}) op_select[2] = RegBRep;
      end
      riscv_instr::VFNDOTPEX_S_H,
      riscv_instr::VFNDOTPEX_S_R_H: begin
        sb_out.fpu_op = fpnew_pkg::SDOTP;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFNDOTPEX_S_R_H}) op_select[2] = RegBRep;
      end
      riscv_instr::VFSUMEX_S_H,
      riscv_instr::VFNSUMEX_S_H: begin
        sb_out.fpu_op = fpnew_pkg::EXVSUM;
        op_select[0] = RegA;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFNSUMEX_S_H}) sb_out.op_mode = 1'b1;
      end
      // [Alternate] Quarter Precision
      riscv_instr::FADD_B: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FSUB_B: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.op_mode = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FMUL_B: begin
        sb_out.fpu_op = fpnew_pkg::MUL;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FDIV_B: begin
        sb_out.fpu_op = fpnew_pkg::DIV;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FSGNJ_B,
      riscv_instr::FSGNJN_B,
      riscv_instr::FSGNJX_B: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FMIN_B,
      riscv_instr::FMAX_B: begin
        sb_out.fpu_op = fpnew_pkg::MINMAX;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FSQRT_B: begin
        sb_out.fpu_op = fpnew_pkg::SQRT;
        op_select[0] = RegA;
        op_select[1] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FMADD_B: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FMSUB_B: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FNMSUB_B: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FNMADD_B: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegC;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::VFSUM_B,
      riscv_instr::VFNSUM_B: begin
        sb_out.fpu_op = fpnew_pkg::VSUM;
        op_select[0] = RegA;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFNSUM_B}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::FMULEX_S_B: begin
        sb_out.fpu_op = fpnew_pkg::MUL;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FMACEX_S_B: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_S_B: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_B_S: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
      end
      riscv_instr::FCVT_D_B: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
      end
      riscv_instr::FCVT_B_D: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
      end
      riscv_instr::FCVT_H_B: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
      end
      riscv_instr::FCVT_B_H: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
      end
      // Vectorial [alternate] Quarter Precision
      riscv_instr::VFADD_B,
      riscv_instr::VFADD_R_B: begin
        sb_out.fpu_op = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFADD_R_B}) op_select[2] = RegBRep;
      end
      riscv_instr::VFSUB_B,
      riscv_instr::VFSUB_R_B: begin
        sb_out.fpu_op  = fpnew_pkg::ADD;
        op_select[1] = RegA;
        op_select[2] = RegB;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSUB_R_B}) op_select[2] = RegBRep;
      end
      riscv_instr::VFMUL_B,
      riscv_instr::VFMUL_R_B: begin
        sb_out.fpu_op = fpnew_pkg::MUL;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMUL_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFDIV_B,
      riscv_instr::VFDIV_R_B: begin
        sb_out.fpu_op = fpnew_pkg::DIV;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFDIV_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFMIN_B,
      riscv_instr::VFMIN_R_B: begin
        sb_out.fpu_op = fpnew_pkg::MINMAX;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RNE;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMIN_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFMAX_B,
      riscv_instr::VFMAX_R_B: begin
        sb_out.fpu_op = fpnew_pkg::MINMAX;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.fpu_rnd_mode = fpnew_pkg::RTZ;
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMAX_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSQRT_B: begin
        sb_out.fpu_op = fpnew_pkg::SQRT;
        op_select[0] = RegA;
        op_select[1] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
      end
      riscv_instr::VFMAC_B,
      riscv_instr::VFMAC_R_B: begin
        sb_out.fpu_op = fpnew_pkg::FMADD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMAC_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFMRE_B,
      riscv_instr::VFMRE_R_B: begin
        sb_out.fpu_op = fpnew_pkg::FNMSUB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFMRE_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJ_B,
      riscv_instr::VFSGNJ_R_B: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RNE;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSGNJ_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJN_B,
      riscv_instr::VFSGNJN_R_B: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RTZ;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSGNJN_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFSGNJX_B,
      riscv_instr::VFSGNJX_R_B: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = RegA;
        op_select[1] = RegB;
        sb_out.fpu_rnd_mode = fpnew_pkg::RDN;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFSGNJX_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFCPKA_B_S,
      riscv_instr::VFCPKB_B_S: begin
        sb_out.fpu_op = fpnew_pkg::CPKAB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCPKB_B_S}) sb_out.op_mode = 1;
      end
      riscv_instr::VFCPKC_B_S,
      riscv_instr::VFCPKD_B_S: begin
        sb_out.fpu_op = fpnew_pkg::CPKCD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCPKD_B_S}) sb_out.op_mode = 1;
      end
     riscv_instr::VFCPKA_B_D,
      riscv_instr::VFCPKB_B_D: begin
        sb_out.fpu_op = fpnew_pkg::CPKAB;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCPKB_B_D}) sb_out.op_mode = 1;
      end
      riscv_instr::VFCPKC_B_D,
      riscv_instr::VFCPKD_B_D: begin
        sb_out.fpu_op = fpnew_pkg::CPKCD;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCPKD_B_D}) sb_out.op_mode = 1;
      end
      riscv_instr::VFCVT_S_B,
      riscv_instr::VFCVTU_S_B: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVTU_S_B}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_B_S,
      riscv_instr::VFCVTU_B_S: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVTU_B_S}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_H_H,
      riscv_instr::VFCVTU_H_H: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVTU_H_H}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_H_B,
      riscv_instr::VFCVTU_H_B: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVTU_H_B}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_B_H,
      riscv_instr::VFCVTU_B_H: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVTU_B_H}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::VFCVT_B_B,
      riscv_instr::VFCVTU_B_B: begin
        sb_out.fpu_op = fpnew_pkg::F2F;
        op_select[0] = RegA;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVTU_B_B}) sb_out.op_mode = 1'b1;
      end
      riscv_instr::VFDOTPEX_H_B,
      riscv_instr::VFDOTPEX_H_R_B: begin
        sb_out.fpu_op = fpnew_pkg::SDOTP;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFDOTPEX_H_R_B}) op_select[2] = RegBRep;
      end
      riscv_instr::VFNDOTPEX_H_B,
      riscv_instr::VFNDOTPEX_H_R_B: begin
        sb_out.fpu_op = fpnew_pkg::SDOTP;
        op_select[0] = RegA;
        op_select[1] = RegB;
        op_select[2] = RegDest;
        sb_out.op_mode      = 1'b1;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFNDOTPEX_H_R_B}) op_select[2] = RegBRep;
      end
      riscv_instr::VFSUMEX_H_B,
      riscv_instr::VFNSUMEX_H_B: begin
        sb_out.fpu_op = fpnew_pkg::EXVSUM;
        op_select[0] = RegA;
        op_select[2] = RegDest;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFNSUMEX_H_B}) sb_out.op_mode = 1'b1;
      end
      // -------------------
      // From float to int
      // -------------------
      // Single Precision Floating-Point
      riscv_instr::FLE_S,
      riscv_instr::FLT_S,
      riscv_instr::FEQ_S: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::FCLASS_S: begin
        sb_out.fpu_op = fpnew_pkg::CLASSIFY;
        op_select[0]   = RegA;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::FCVT_W_S,
      riscv_instr::FCVT_WU_S: begin
        sb_out.fpu_op = fpnew_pkg::F2I;
        op_select[0]   = RegA;
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::FCVT_WU_S}) sb_out.op_mode = 1'b1; // unsigned
      end
      riscv_instr::FMV_X_W: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.op_mode        = 1'b1; // sign-extend result
        op_select[0]   = RegA;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      // Vectorial Single Precision
      riscv_instr::VFEQ_S,
      riscv_instr::VFEQ_R_S: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RDN;
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFEQ_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFNE_S,
      riscv_instr::VFNE_R_S: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RDN;
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.op_mode        = 1'b1;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFNE_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFLT_S,
      riscv_instr::VFLT_R_S: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RTZ;
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFLT_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFGE_S,
      riscv_instr::VFGE_R_S: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RTZ;
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.op_mode        = 1'b1;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFGE_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFLE_S,
      riscv_instr::VFLE_R_S: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFLE_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFGT_S,
      riscv_instr::VFGT_R_S: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.op_mode        = 1'b1;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFGT_R_S}) op_select[1] = RegBRep;
      end
      riscv_instr::VFCLASS_S: begin
        sb_out.fpu_op = fpnew_pkg::CLASSIFY;
        op_select[0]   = RegA;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP32;
        sb_out.dst_fmt        = fpnew_pkg::FP32;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      // Double Precision Floating-Point
      riscv_instr::FLE_D,
      riscv_instr::FLT_D,
      riscv_instr::FEQ_D: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.src_fmt        = fpnew_pkg::FP64;
        sb_out.dst_fmt        = fpnew_pkg::FP64;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::FCLASS_D: begin
        sb_out.fpu_op = fpnew_pkg::CLASSIFY;
        op_select[0]   = RegA;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP64;
        sb_out.dst_fmt        = fpnew_pkg::FP64;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::FCVT_W_D,
      riscv_instr::FCVT_WU_D: begin
        sb_out.fpu_op = fpnew_pkg::F2I;
        op_select[0]   = RegA;
        sb_out.src_fmt        = fpnew_pkg::FP64;
        sb_out.dst_fmt        = fpnew_pkg::FP64;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::FCVT_WU_D}) sb_out.op_mode = 1'b1; // unsigned
      end
      riscv_instr::FMV_X_D: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt        = fpnew_pkg::FP64;
        sb_out.dst_fmt        = fpnew_pkg::FP64;
        sb_out.op_mode        = 1'b1; // sign-extend result
        op_select[0]   = RegA;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      // [Alternate] Half Precision Floating-Point
      riscv_instr::FLE_H,
      riscv_instr::FLT_H,
      riscv_instr::FEQ_H: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::FCLASS_H: begin
        sb_out.fpu_op = fpnew_pkg::CLASSIFY;
        op_select[0]   = RegA;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::FCVT_W_H,
      riscv_instr::FCVT_WU_H: begin
        sb_out.fpu_op = fpnew_pkg::F2I;
        op_select[0]   = RegA;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::FCVT_WU_H}) sb_out.op_mode = 1'b1; // unsigned
      end
      riscv_instr::FMV_X_H: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.op_mode        = 1'b1; // sign-extend result
        op_select[0]   = RegA;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      // Vectorial [alternate] Half Precision
      riscv_instr::VFEQ_H,
      riscv_instr::VFEQ_R_H: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RDN;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFEQ_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFNE_H,
      riscv_instr::VFNE_R_H: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RDN;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.op_mode        = 1'b1;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFNE_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFLT_H,
      riscv_instr::VFLT_R_H: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RTZ;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFLT_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFGE_H,
      riscv_instr::VFGE_R_H: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RTZ;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.op_mode        = 1'b1;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFGE_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFLE_H,
      riscv_instr::VFLE_R_H: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFLE_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFGT_H,
      riscv_instr::VFGT_R_H: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.op_mode        = 1'b1;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFGT_R_H}) op_select[1] = RegBRep;
      end
      riscv_instr::VFCLASS_H: begin
        sb_out.fpu_op = fpnew_pkg::CLASSIFY;
        op_select[0]   = RegA;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::VFMV_X_H: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.op_mode        = 1'b1; // sign-extend result
        op_select[0]   = RegA;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::VFCVT_X_H,
      riscv_instr::VFCVT_XU_H: begin
        sb_out.fpu_op = fpnew_pkg::F2I;
        op_select[0]   = RegA;
        sb_out.src_fmt        = fpnew_pkg::FP16;
        sb_out.dst_fmt        = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP16ALT;
        end
        sb_out.int_fmt        = fpnew_pkg::INT16;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        sb_out.set_dyn_rm     = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVT_XU_H}) sb_out.op_mode = 1'b1; // upper
      end
      // [Alternate] Quarter Precision Floating-Point
      riscv_instr::FLE_B,
      riscv_instr::FLT_B,
      riscv_instr::FEQ_B: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::FCLASS_B: begin
        sb_out.fpu_op = fpnew_pkg::CLASSIFY;
        op_select[0]   = RegA;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::FCVT_W_B,
      riscv_instr::FCVT_WU_B: begin
        sb_out.fpu_op = fpnew_pkg::F2I;
        op_select[0]   = RegA;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::FCVT_WU_B}) sb_out.op_mode = 1'b1; // unsigned
      end
      riscv_instr::FMV_X_B: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.op_mode        = 1'b1; // sign-extend result
        op_select[0]   = RegA;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      // Vectorial Quarter Precision
      riscv_instr::VFEQ_B,
      riscv_instr::VFEQ_R_B: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RDN;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFEQ_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFNE_B,
      riscv_instr::VFNE_R_B: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RDN;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.op_mode        = 1'b1;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFNE_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFLT_B,
      riscv_instr::VFLT_R_B: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RTZ;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFLT_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFGE_B,
      riscv_instr::VFGE_R_B: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RTZ;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.op_mode        = 1'b1;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFGE_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFLE_B,
      riscv_instr::VFLE_R_B: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFLE_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFGT_B,
      riscv_instr::VFGT_R_B: begin
        sb_out.fpu_op = fpnew_pkg::CMP;
        op_select[0]   = RegA;
        op_select[1]   = RegB;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.op_mode        = 1'b1;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        if (seq_out_q.data_op inside {riscv_instr::VFGT_R_B}) op_select[1] = RegBRep;
      end
      riscv_instr::VFCLASS_B: begin
        sb_out.fpu_op = fpnew_pkg::CLASSIFY;
        op_select[0]   = RegA;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RNE;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::VFMV_X_B: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        sb_out.fpu_rnd_mode   = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.op_mode        = 1'b1; // sign-extend result
        op_select[0]   = RegA;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
      end
      riscv_instr::VFCVT_X_B,
      riscv_instr::VFCVT_XU_B: begin
        sb_out.fpu_op = fpnew_pkg::F2I;
        op_select[0]   = RegA;
        sb_out.src_fmt        = fpnew_pkg::FP8;
        sb_out.dst_fmt        = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.src == 1'b1) begin
          sb_out.src_fmt      = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt      = fpnew_pkg::FP8ALT;
        end
        sb_out.int_fmt        = fpnew_pkg::INT8;
        sb_out.vectorial_op   = 1'b1;
        sb_out.fpu_tag_in.acc = 1'b1;
        rd_is_fp       = 1'b0;
        sb_out.set_dyn_rm     = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVT_XU_B}) sb_out.op_mode = 1'b1; // upper
      end
      // -------------------
      // From int to float
      // -------------------
      // Single Precision Floating-Point
      riscv_instr::FMV_W_X: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = AccBus;
        sb_out.fpu_rnd_mode = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt      = fpnew_pkg::FP32;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
      end
      riscv_instr::FCVT_S_W,
      riscv_instr::FCVT_S_WU: begin
        sb_out.fpu_op = fpnew_pkg::I2F;
        op_select[0] = AccBus;
        sb_out.dst_fmt      = fpnew_pkg::FP32;
        if (seq_out_q.data_op inside {riscv_instr::FCVT_S_WU}) sb_out.op_mode = 1'b1; // unsigned
      end
      // Double Precision Floating-Point
      riscv_instr::FCVT_D_W,
      riscv_instr::FCVT_D_WU: begin
        sb_out.fpu_op = fpnew_pkg::I2F;
        op_select[0] = AccBus;
        sb_out.src_fmt      = fpnew_pkg::FP64;
        sb_out.dst_fmt      = fpnew_pkg::FP64;
        if (seq_out_q.data_op inside {riscv_instr::FCVT_D_WU}) sb_out.op_mode = 1'b1; // unsigned
      end
      // [Alternate] Half Precision Floating-Point
      riscv_instr::FMV_H_X: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = AccBus;
        sb_out.fpu_rnd_mode = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
      end
      riscv_instr::FCVT_H_W,
      riscv_instr::FCVT_H_WU: begin
        sb_out.fpu_op = fpnew_pkg::I2F;
        op_select[0] = AccBus;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        if (seq_out_q.data_op inside {riscv_instr::FCVT_H_WU}) sb_out.op_mode = 1'b1; // unsigned
      end
      // Vectorial Half Precision Floating-Point
      riscv_instr::VFMV_H_X: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = AccBus;
        sb_out.fpu_rnd_mode = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.vectorial_op = 1'b1;
      end
      riscv_instr::VFCVT_H_X,
      riscv_instr::VFCVT_H_XU: begin
        sb_out.fpu_op = fpnew_pkg::I2F;
        op_select[0] = AccBus;
        sb_out.src_fmt      = fpnew_pkg::FP16;
        sb_out.dst_fmt      = fpnew_pkg::FP16;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP16ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP16ALT;
        end
        sb_out.int_fmt      = fpnew_pkg::INT16;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVT_H_XU}) sb_out.op_mode = 1'b1; // upper
      end
      // [Alternate] Quarter Precision Floating-Point
      riscv_instr::FMV_B_X: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = AccBus;
        sb_out.fpu_rnd_mode = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (fpu_fmt_mode_i.dst == 1'b1) begin
          sb_out.src_fmt    = fpnew_pkg::FP8ALT;
          sb_out.dst_fmt    = fpnew_pkg::FP8ALT;
        end
      end
      riscv_instr::FCVT_B_W,
      riscv_instr::FCVT_B_WU: begin
        sb_out.fpu_op = fpnew_pkg::I2F;
        op_select[0] = AccBus;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        if (seq_out_q.data_op inside {riscv_instr::FCVT_B_WU}) sb_out.op_mode = 1'b1; // unsigned
      end
      // Vectorial Quarter Precision Floating-Point
      riscv_instr::VFMV_B_X: begin
        sb_out.fpu_op = fpnew_pkg::SGNJ;
        op_select[0] = AccBus;
        sb_out.fpu_rnd_mode = fpnew_pkg::RUP; // passthrough without checking nan-box
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        sb_out.vectorial_op = 1'b1;
      end
      riscv_instr::VFCVT_B_X,
      riscv_instr::VFCVT_B_XU: begin
        sb_out.fpu_op = fpnew_pkg::I2F;
        op_select[0] = AccBus;
        sb_out.src_fmt      = fpnew_pkg::FP8;
        sb_out.dst_fmt      = fpnew_pkg::FP8;
        sb_out.int_fmt      = fpnew_pkg::INT8;
        sb_out.vectorial_op = 1'b1;
        sb_out.set_dyn_rm   = 1'b1;
        if (seq_out_q.data_op inside {riscv_instr::VFCVT_B_XU}) sb_out.op_mode = 1'b1; // upper
      end
      // -------------
      // Load / Store
      // -------------
      // Single Precision Floating-Point
      riscv_instr::FLW: begin
        sb_out.is_load = 1'b1;
        use_fpu = 1'b0;
      end
      riscv_instr::FSW: begin
        sb_out.is_store = 1'b1;
        op_select[1] = RegB;
        use_fpu = 1'b0;
        rd_is_fp = 1'b0;
      end
      // Double Precision Floating-Point
      riscv_instr::FLD: begin
        sb_out.is_load = 1'b1;
        sb_out.ls_size = DoubleWord;
        use_fpu = 1'b0;
      end
      riscv_instr::FSD: begin
        sb_out.is_store = 1'b1;
        op_select[1] = RegB;
        sb_out.ls_size = DoubleWord;
        use_fpu = 1'b0;
        rd_is_fp = 1'b0;
      end
      // [Alternate] Half Precision Floating-Point
      riscv_instr::FLH: begin
        sb_out.is_load = 1'b1;
        sb_out.ls_size = HalfWord;
        use_fpu = 1'b0;
      end
      riscv_instr::FSH: begin
        sb_out.is_store = 1'b1;
        op_select[1] = RegB;
        sb_out.ls_size = HalfWord;
        use_fpu = 1'b0;
        rd_is_fp = 1'b0;
      end
      // [Alternate] Quarter Precision Floating-Point
      riscv_instr::FLB: begin
        sb_out.is_load = 1'b1;
        sb_out.ls_size = Byte;
        use_fpu = 1'b0;
      end
      riscv_instr::FSB: begin
        sb_out.is_store = 1'b1;
        op_select[1] = RegB;
        sb_out.ls_size = Byte;
        use_fpu = 1'b0;
        rd_is_fp = 1'b0;
      end
      // -------------
      // CSR Handling
      // -------------
      // Set or clear corresponding CSR
      riscv_instr::CSRRSI: begin
        use_fpu = 1'b0;
        rd_is_fp = 1'b0;
        csr_instr = 1'b1;
        // ssr_active_d |= rs1[0];
      end
      riscv_instr::CSRRCI: begin
        use_fpu = 1'b0;
        rd_is_fp = 1'b0;
        csr_instr = 1'b1;
        // ssr_active_d &= ~rs1[0];
      end
      default: begin
        use_fpu = 1'b0;
        acc_resp_o.error = 1'b1;
        rd_is_fp = 1'b0;
      end
    endcase
    // fix round mode for vectors and fp16alt
    if (sb_out.set_dyn_rm) sb_out.fpu_rnd_mode = fpu_rnd_mode_i;
    // check if sb_out.src_fmt or sb_out.dst_fmt is acutually the alternate version
    // single-format float operations ignore fpu_fmt_mode_i.src
    // reason: for performance reasons when mixing expanding and non-expanding operations
    if (sb_out.src_fmt == fpnew_pkg::FP16 && fpu_fmt_mode_i.src == 1'b1) sb_out.src_fmt = fpnew_pkg::FP16ALT;
    if (sb_out.dst_fmt == fpnew_pkg::FP16 && fpu_fmt_mode_i.dst == 1'b1) sb_out.dst_fmt = fpnew_pkg::FP16ALT;
    if (sb_out.src_fmt == fpnew_pkg::FP8 && fpu_fmt_mode_i.src == 1'b1) sb_out.src_fmt = fpnew_pkg::FP8ALT;
    if (sb_out.dst_fmt == fpnew_pkg::FP8 && fpu_fmt_mode_i.dst == 1'b1) sb_out.dst_fmt = fpnew_pkg::FP8ALT;
  end

  always_comb begin
    fpr_raddr[0] = rs1;
    fpr_raddr[1] = rs2;
    fpr_raddr[2] = rs3;

    unique case (op_select[1])
      RegA: begin
        fpr_raddr[1] = rs1;
      end
      default:;
    endcase

    unique case (op_select[2])
      RegB,
      RegBRep: begin
        fpr_raddr[2] = rs2;
      end
      RegDest: begin
        fpr_raddr[2] = rd;
      end
      default:;
    endcase
  end

  // Calculate register addresses
  assign op_reg_names = {fpr_raddr[0], fpr_raddr[1], fpr_raddr[2], rd};
  for (genvar i = 0; i < 4; i++) begin
    always_comb begin
      case (op_reg_names[i][4:3])
        0: sb_out.fpr_bank_raddr[i] = {cfg_offsets_q[0], op_reg_names[i][2:0]};
        1: sb_out.fpr_bank_raddr[i] = {cfg_offsets_q[1], op_reg_names[i][2:0]};
        2: sb_out.fpr_bank_raddr[i] = {cfg_offsets_q[2], op_reg_names[i][2:0]};
        3: sb_out.fpr_bank_raddr[i] = {cfg_offsets_q[3], op_reg_names[i][2:0]};
        default:;
      endcase
    end
  end

  logic [ScoreboardDepth-1:0] sb_pop_index;
  logic sb_pop_valid;

  // Scoreboard to keep track of addresses currently in pipeline
  stitch_sb #(
    .AddrWidth(TotalAddrWidth),
    .Depth(ScoreboardDepth),
    .NumTestAddrs(4)
  ) i_sb (
    .clk_i,
    .rst_i,
    .push_rd_addr_i(sb_out.fpr_bank_raddr[3]),
    .push_valid_i(seq_out_valid_q & seq_out_ready_q & rd_is_fp),
    .entry_index_o(rd_sb_index),
    .pop_index_i(sb_pop_index),
    .pop_valid_i(sb_pop_valid),
    .test_rd_addr_i(sb_out.fpr_bank_raddr),
    .test_addr_present_o(sb_hit),
    .full_o(/*unused*/)
  );

  // avoid loading duplicated registers or registers that
  // refer to an address currently in the pipeline
  logic [2:0] fpr_req_enable;
  logic [2:0][1:0] fpr_reg_coallesce;
  always_comb begin
    fpr_reg_coallesce[0] = 0;
    fpr_reg_coallesce[1] = 1;
    fpr_reg_coallesce[2] = 2;
    fpr_req_enable[0] = (op_select[0] != AccBus) & (op_select[0] != None);
    fpr_req_enable[1] = (op_select[1] != AccBus) & (op_select[1] != None);
    fpr_req_enable[2] = (op_select[2] != AccBus) & (op_select[2] != None);
    if (sb_hit[0]) begin
      fpr_reg_coallesce[0] = 3;
      fpr_req_enable[0] = 0;
    end
    if (sb_hit[1]) begin
      fpr_reg_coallesce[1] = 3;
      fpr_req_enable[1] = 0;
    end else if (op_reg_names[1] == op_reg_names[0]) begin
      fpr_reg_coallesce[1] = 0;
      fpr_req_enable[1] = 0;
    end
    if (sb_hit[2]) begin
      fpr_reg_coallesce[2] = 3;
      fpr_req_enable[2] = 0;
    end else if (op_reg_names[2] == op_reg_names[0]) begin
      fpr_reg_coallesce[2] = 0;
      fpr_req_enable[2] = 0;
    end else if (op_reg_names[2] == op_reg_names[1]) begin
      fpr_reg_coallesce[2] = 1;
      fpr_req_enable[2] = 0;
    end
  end

  // Scoreboard/Operand Valid
  // Track floating point destination registers
  always_comb begin
    // reset the value if we are committing the register
    if (fpr_we) sb_d[fpr_waddr] = 1'b0;
    // don't track any dependencies for SSRs if enabled
    // if (ssr_active_q) begin
    //   for (int i = 0; i < NumSsrs; i++) sb_d[SsrRegs[i]] = 1'b0;
    // end
  end

  // optional spill register
  spill_register  #(
    .T      ( sb_dec_out_t     ),
    .Bypass ( !RegisterDecoder )
  ) i_spill_register_acc (
    .clk_i   ,
    .rst_ni  ( ~rst_i         ),
    .valid_i ( sb_out_valid   ),
    .ready_o ( sb_out_ready   ),
    .data_i  ( sb_out         ),
    .valid_o ( sb_out_valid_q ),
    .ready_i ( sb_out_ready_q ),
    .data_o  ( sb_out_q       )
  );
  
  // -------------
  // FPU Virtual Register File
  // -------------
  typedef struct packed {
    logic [2:0]                     fpr_req_enable;
    logic [2:0][1:0]                fpr_reg_coallesce;
    fpnew_pkg::operation_e          fpu_op;
    fpnew_pkg::roundmode_e          fpu_rnd_mode;
    fpnew_pkg::fp_format_e          src_fmt, dst_fmt;
    fpnew_pkg::int_format_e         int_fmt;
    logic                           vectorial_op;
    logic                           set_dyn_rm;
    tag_t                           fpu_tag_in;
    logic                           is_load;
    logic                           is_store;
    ls_size_e                       ls_size;
  } vfpr_out_t;

  logic vfpr_out_valid, vfpr_out_valid_q;
  logic vfpr_out_ready, vfpr_out_ready_q;
  vfpr_out_t vfpr_out, vfpr_out_q;

  assign vfpr_out.fpr_req_enable = sb_out_q.fpr_req_enable;
  assign vfpr_out.fpr_reg_coallesce = sb_out_q.fpr_reg_coallesce;
  assign vfpr_out.fpu_op = sb_out_q.fpu_op;
  assign vfpr_out.fpu_rnd_mode = sb_out_q.fpu_rnd_mode;
  assign vfpr_out.src_fmt = sb_out_q.src_fmt;
  assign vfpr_out.int_fmt = sb_out_q.int_fmt;
  assign vfpr_out.vectorial_op = sb_out_q.vectorial_op;
  assign vfpr_out.set_dyn_rm = sb_out_q.set_dyn_rm;
  assign vfpr_out.fpu_tag_in = sb_out_q.fpu_tag_in;
  assign vfpr_out.is_load = sb_out_q.is_load;
  assign vfpr_out.is_store = sb_out_q.is_store;
  assign vfpr_out.ls_size = sb_out_q.ls_size;

  vfpr_req_t [3:0] vfpr_reqs;
  vfpr_rsp_t [3:0] vfpr_rsps;

  for (genvar i = 0; i < 3; i++) begin
    assign vfpr_reqs[i].q.addr = sb_out_q.fpr_bank_raddr[i];
    assign vfpr_reqs[i].q.write = '0;
    assign vfpr_reqs[i].q.amo = reqrsp_pkg::AMONone;
    assign vfpr_reqs[i].q.data = '0;
    assign vfpr_reqs[i].q.strb = '1;
    assign vfpr_reqs[i].q.user = '0;
    assign vfpr_reqs[i].q_valid = sb_out_valid_q & sb_out_q.fpr_req_enable[i];
  end

  assign vfpr_reqs[3].q.addr = fpr_waddr;
  assign vfpr_reqs[3].q.write = '1;
  assign vfpr_reqs[3].q.amo = reqrsp_pkg::AMONone;
  assign vfpr_reqs[3].q.data = fpr_wdata;
  assign vfpr_reqs[3].q.strb = '1;
  assign vfpr_reqs[3].q.user = '0;
  assign vfpr_reqs[3].q_valid = fpr_we;

  // request rs1-rs3 from memory
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
    .mem_req_o(data_fast_req_o),
    .mem_rsp_i(data_fast_rsp_i)
  );

  // all register load paths have to be ready so that the next
  // instruction can proceed. Otherwise a bunch of FIFOs are
  // needed to buffer the input
  assign sb_out_ready_q = &({
    vfpr_rsps[0].q_ready, 
    vfpr_rsps[1].q_ready, 
    vfpr_rsps[2].q_ready
  });

  // have to buffer pipeline state until the operands
  // have arrived back. Ideally this happens in 1 cycle
  acc_req_t acc_req_qq;
  logic [3:0][FLEN-1:0] rs_reg_data_q;
  logic [3:0][FLEN-1:0] rs_reg_valid_q;

  spill_register  #(
    .T      ( vfpr_out_t ),
  ) i_spill_register_acc (
    .clk_i   ,
    .rst_ni  ( ~rst_i           ),
    .valid_i ( vfpr_out_valid   ),
    .ready_o ( vfpr_out_ready   ),
    .data_i  ( vfpr_out         ),
    .valid_o ( vfpr_out_valid_q ),
    .ready_i ( vfpr_out_ready_q ),
    .data_o  ( vfpr_out_q       )
  );

  logic [3:0] rs_result_we;
  for (genvar i = 0; i < 3; i++) begin
    rs_result_we[i] = vfpr_rsps[i].p_valid | (~vfpr_out_q.fpr_req_enable[i] & vfpr_out_ready);
  end

  `FFLAR(rs_reg_data_q[0], vfpr_rsps[0].p.data, vfpr_rsps[0].p_valid, '0, clk_i, rst_i)
  `FFLAR(rs_reg_data_q[1], vfpr_rsps[1].p.data, vfpr_rsps[1].p_valid, '0, clk_i, rst_i)
  `FFLAR(rs_reg_data_q[2], vfpr_rsps[2].p.data, vfpr_rsps[2].p_valid, '0, clk_i, rst_i)
  `FFLAR(rs_reg_valid_q[0], vfpr_rsps[0].p_valid, vfpr_rsps[0].p_valid, '0, clk_i, rst_i)
  `FFLAR(rs_reg_valid_q[1], vfpr_rsps[1].p_valid, vfpr_rsps[1].p_valid, '0, clk_i, rst_i)
  `FFLAR(rs_reg_valid_q[2], vfpr_rsps[2].p_valid, vfpr_rsps[2].p_valid, '0, clk_i, rst_i)

  // register collalescing
  always_comb begin

  end

  // Ensure SSR CSR only written on instruction commit
  // assign ssr_active_ena = seq_out_valid_q & seq_out_ready_q;

  // this handles WAW Hazards - Potentially this can be relaxed if necessary
  // at the expense of increased timing pressure
  assign dst_ready = ~(rd_is_fp & |(sb_hit));

  // check that either:
  // 1. The FPU and all operands are ready
  // 2. The LSU request can be handled
  // 3. The regfile operand is ready
  assign fpu_in_valid = use_fpu & seq_out_valid_q & (&op_ready) & dst_ready;
                                      // FPU ready
  assign seq_out_ready_q = dst_ready & ((fpu_in_ready & fpu_in_valid)
                                      // Load/Store
                                      | (lsu_qvalid & lsu_qready)
                                      | csr_instr
                                      // Direct Reg Wriopte
                                      | (seq_out_valid_q && result_select == ResAccBus));

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

  // Regfile offsets and strides
  for (genvar i = 0; i < 4; i++) begin
    always_comb begin
      cfg_offsets_d[i] = cfg_offsets_q[i];
      cfg_strides_d[i] = cfg_strides_q[i];

      // the MSB determines if this is a write to offset or stride
      // the LSBs determine which offsets/strides to modify
      if (cfg_write_i & ~cfg_word_i[4] & cfg_word_i[i]) begin
        cfg_offsets_d[i] = cfg_wdata_i[RegGroupsPerBank-1:0];
      end

      if (cfg_write_i & cfg_word_i[4] & cfg_word_i[i]) begin
        cfg_strides_d[i] = cfg_wdata_i[RegGroupsPerBank-1:0];
      end
    end
  end

  // perhaps at some point there will be a way to read these values
  // through a seperate interface
  assign cfg_rdata_o = '0;
  assign cfg_wready_o = 0'b1;

  // Determine whether destination register is SSR
  // logic is_rd_ssr;
  // always_comb begin
  //   is_rd_ssr = 1'b0;
  //   for (int s = 0; s < NumSsrs; s++)
  //     is_rd_ssr |= (SsrRegs[s] == rd);
  // end

  // snitch_regfile #(
  //   .DATA_WIDTH     ( FLEN ),
  //   .NR_READ_PORTS  ( 3    ),
  //   .NR_WRITE_PORTS ( 1    ),
  //   .ZERO_REG_ZERO  ( 0    ),
  //   .ADDR_WIDTH     ( 5    )
  // ) i_ff_regfile (
  //   .clk_i,
  //   .raddr_i   ( fpr_raddr ),
  //   .rdata_o   ( fpr_rdata ),
  //   .waddr_i   ( fpr_waddr ),
  //   .wdata_i   ( fpr_wdata ),
  //   .we_i      ( fpr_we    )
  // );

  // ----------------------
  // Operand Select
  // ----------------------
  logic [2:0][FLEN-1:0] acc_qdata;
  assign acc_qdata = {seq_out_q.data_argc, seq_out_q.data_argb, seq_out_q.data_arga};

  for (genvar i = 0; i < 3; i++) begin: gen_operand_select
    // logic is_raddr_ssr;
    // always_comb begin
    //   is_raddr_ssr = 1'b0;
    //   for (int s = 0; s < NumSsrs; s++)
    //     is_raddr_ssr |= (SsrRegs[s] == fpr_raddr[i]);
    // end
    always_comb begin
      // ssr_rvalid_o[i] = 1'b0;
      unique case (op_select[i])
        None: begin
          op[i] = '1;
          op_ready[i] = 1'b1;
        end
        AccBus: begin
          op[i] = acc_qdata[i];
          op_ready[i] = seq_out_valid_q;
        end
        // Scoreboard or SSR
        RegA, RegB, RegBRep, RegC, RegDest: begin
          // map register 0 and 1 to SSRs
          // ssr_rvalid_o[i] = ssr_active_q & is_raddr_ssr;
          // op[i] = ssr_rvalid_o[i] ? ssr_rdata_i[i] : fpr_rdata[i];
          op[i] = fpr_rdata[i];
          // the operand is ready if it is not marked in the scoreboard
          // and in case of it being an SSR it need to be ready as well
          // op_ready[i] = ~sb_q[fpr_raddr[i]] & (~ssr_rvalid_o[i] | ssr_rready_i[i]);
          op_ready[i] = ~sb_q[fpr_raddr[i]];
          // Replicate if needed
          if (op_select[i] == RegBRep) begin
            unique case (sb_out.src_fmt)
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
          op_ready[i] = 1'b1;
        end
      endcase
    end
  end

  // ----------------------
  // Floating Point Unit
  // ----------------------
  stitch_fpu #(
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
    .rnd_mode_i     ( sb_out.fpu_rnd_mode  ),
    .op_i           ( sb_out.fpu_op        ),
    .op_mod_i       ( sb_out.op_mode       ), // Sign of operand?
    .src_fmt_i      ( sb_out.src_fmt       ),
    .dst_fmt_i      ( sb_out.dst_fmt       ),
    .int_fmt_i      ( sb_out.int_fmt       ),
    .vectorial_op_i ( sb_out.vectorial_op  ),
    .tag_i          ( fpu_tag_in    ),
    .in_valid_i     ( fpu_in_valid  ),
    .in_ready_o     ( fpu_in_ready  ),
    .result_o       ( fpu_result    ),
    .status_o       ( fpu_status_o  ),
    .tag_o          ( fpu_tag_out   ),
    .out_valid_o    ( fpu_out_valid ),
    .out_ready_i    ( fpu_out_ready )
  );

  // assign ssr_waddr_o = fpr_waddr;
  // assign ssr_wdata_o = fpr_wdata;
  logic [63:0] nan_boxed_arga;
  assign nan_boxed_arga = {{32{1'b1}}, seq_out_q.data_arga[31:0]};

  // Arbitrate Register File Write Port
  always_comb begin
    fpr_we = 1'b0;
    fpr_waddr = '0;
    fpr_wdata = '0;
    fpr_wvalid = 1'b0;
    lsu_pready = 1'b0;
    fpr_wready = 1'b1;
    // ssr_wvalid_o = 1'b0;
    // ssr_wdone_o = 1'b1;
    // the accelerator master wants to write
    if (seq_out_valid_q && result_select == ResAccBus) begin
      fpr_we = 1'b1;
      // NaN-Box the value
      fpr_wdata = nan_boxed_arga[FLEN-1:0];
      fpr_waddr = rd;
      fpr_wvalid = 1'b1;
      fpr_wready = 1'b0;
    end else if (fpu_out_valid && !fpu_tag_out.acc) begin
      fpr_we = 1'b1;
      // if (fpu_tag_out.ssr) begin
      //   // ssr_wvalid_o = 1'b1;
      //   // stall write-back to SSR
      //   if (!ssr_wready_i) begin
      //     fpr_wready = 1'b0;
      //     fpr_we = 1'b0;
      //   end else begin
      //     ssr_wdone_o = 1'b1;
      //   end
      // end
      fpr_wdata = fpu_result;
      fpr_waddr = fpu_tag_out.rd;
      fpr_wvalid = 1'b1;
    end else if (lsu_pvalid) begin
      lsu_pready = 1'b1;
      fpr_we = 1'b1;
      fpr_wdata = ld_result;
      fpr_waddr = lsu_rd;
      fpr_wvalid = 1'b1;
      fpr_wready = 1'b0;
    end
  end

  // ----------------------
  // Load/Store Unit
  // ----------------------
  assign lsu_qvalid = seq_out_valid_q & (&op_ready) & (sb_out.is_load | sb_out.is_store);

  snitch_lsu #(
    .AddrWidth (AddrWidth),
    .DataWidth (DataWidth),
    .dreq_t (dreq_t),
    .drsp_t (drsp_t),
    .tag_t (logic [4:0]),
    .NumOutstandingMem (NumFPOutstandingMem),
    .NumOutstandingLoads (NumFPOutstandingLoads),
    .NaNBox (1'b1)
  ) i_snitch_lsu (
    .clk_i (clk_i),
    .rst_i (rst_i),
    .lsu_qtag_i (rd),
    lsu_qwrite_i (is_store),
    .lsu_qsigned_i (1'b1), // all floating point loads are signed
    .lsu_qaddr_i (seq_out_q.data_argc[AddrWidth-1:0]),
    .lsu_qdata_i (op[1]),
    lsu_qsize_i (ls_size),
    .lsu_qamo_i (reqrsp_pkg::AMONone),
    .lsu_qvalid_i (lsu_qvalid),
    .lsu_qready_o (lsu_qready),
    .lsu_pdata_o (ld_result),
    .lsu_ptag_o (lsu_rd),
    .lsu_perror_o (), // ignored for the moment
    .lsu_pvalid_o (lsu_pvalid),
    .lsu_pready_i (lsu_pready),
    .lsu_empty_o (/* unused */),
    .data_req_o(data_slow_req_o),
    .data_rsp_i(data_slow_rsp_i)
  );

  // SSRs
  // for (genvar i = 0; i < 3; i++) assign ssr_rdone_o[i] = ssr_rvalid_o[i] & seq_out_ready_q;
  // assign ssr_raddr_o = fpr_raddr;

  // Counter pipeline.
  logic issue_fpu, issue_core_to_fpu, issue_fpu_seq;
  `FFAR(issue_fpu, fpu_in_valid & fpu_in_ready, 1'b0, clk_i, rst_i)
  `FFAR(issue_core_to_fpu, acc_req_valid_i & acc_req_ready_o, 1'b0, clk_i, rst_i)
  `FFAR(issue_fpu_seq, seq_out_valid & seq_out_ready, 1'b0, clk_i, rst_i)

  always_comb begin
    core_events_o = '0;
    core_events_o.issue_fpu = issue_fpu;
    core_events_o.issue_core_to_fpu = issue_core_to_fpu;
    core_events_o.issue_fpu_seq = issue_fpu_seq;
  end

  // Tracer
  // pragma translate_off
  assign trace_port_o.source       = snitch_pkg::SrcFpu;
  assign trace_port_o.acc_q_hs     = (seq_out_valid_q  && seq_out_ready_q );
  assign trace_port_o.fpu_out_hs   = (fpu_out_valid && fpu_out_ready );
  assign trace_port_o.lsu_q_hs     = (lsu_qvalid    && lsu_qready    );
  assign trace_port_o.op_in        = seq_out_q.data_op;
  assign trace_port_o.rs1          = rs1;
  assign trace_port_o.rs2          = rs2;
  assign trace_port_o.rs3          = rs3;
  assign trace_port_o.rd           = rd;
  assign trace_port_o.op_sel_0     = op_select[0];
  assign trace_port_o.op_sel_1     = op_select[1];
  assign trace_port_o.op_sel_2     = op_select[2];
  assign trace_port_o.sb_out.src_fmt      = sb_out.src_fmt;
  assign trace_port_o.sb_out.dst_fmt      = sb_out.dst_fmt;
  assign trace_port_o.sb_out.int_fmt      = sb_out.int_fmt;
  assign trace_port_o.acc_qdata_0  = acc_qdata[0];
  assign trace_port_o.acc_qdata_1  = acc_qdata[1];
  assign trace_port_o.acc_qdata_2  = acc_qdata[2];
  assign trace_port_o.op_0         = op[0];
  assign trace_port_o.op_1         = op[1];
  assign trace_port_o.op_2         = op[2];
  assign trace_port_o.use_fpu      = use_fpu;
  assign trace_port_o.fpu_in_rd    = fpu_tag_in.rd;
  assign trace_port_o.fpu_in_acc   = sb_out.fpu_tag_in.acc;
  sb_out.assign trace_port_o.ls_size      = ls_size;
  sb_out.assign trace_port_o.is_load      = is_load;
  sb_out.assign trace_port_o.is_store     = is_store;
  assign trace_port_o.lsu_qaddr    = i_snitch_lsu.lsu_qaddr_i;
  assign trace_port_o.lsu_rd       = lsu_rd;
  assign trace_port_o.acc_wb_ready = (result_select == ResAccBus);
  assign trace_port_o.fpu_out_acc  = fpu_tag_out.acc;
  assign trace_port_o.fpr_waddr    = fpr_waddr[0];
  assign trace_port_o.fpr_wdata    = fpr_wdata[0];
  assign trace_port_o.fpr_we       = fpr_we[0];
  // pragma translate_on

  /// Assertions
  `ASSERT(RegWriteKnown, fpr_we |-> !$isunknown(fpr_wdata), clk_i, rst_i)
endmodule
