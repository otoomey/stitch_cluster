`include "common_cells/registers.svh"
`include "common_cells/assertions.svh"

// Floating Point Subsystem
// initialises a wide FPU 
module stitch_fp_ss import snitch_pkg::*; #(
    parameter int unsigned  AddrWidth = 0,
    parameter int unsigned  BankGroupAddrWidth = 0,
    parameter int unsigned  DataWidth = 0,
    parameter int unsigned  NumFPOutstandingLoads = 0,
    parameter int unsigned  NumFPOutstandingMem = 0,
    parameter int unsigned  NumFPUSequencerInstr = 0,
    parameter int unsigned  ScoreboardDepth = 0,
    parameter int unsigned  FLEN = DataWidth,
    parameter bit           RegisterSequencer = 0,
    parameter bit           RegisterDecoder   = 0,
    parameter bit           RegisterFPUIn     = 0,
    parameter bit           RegisterFPUOut    = 0,
    parameter type          tcdm_req_t = logic,
    parameter type          tcdm_rsp_t = logic,
    parameter type          bank_req_t = logic,
    parameter type          bank_rsp_t = logic,
    parameter type          acc_req_t = logic,
    parameter type          acc_resp_t = logic,
    parameter bit           RVF = 1,
    parameter bit           RVD = 1,
    parameter bit           XF16 = 0,
    parameter bit           XF16ALT = 0,
    parameter bit           XF8 = 0,
    parameter bit           XF8ALT = 0,
    parameter bit           XFVEC = 0,
    parameter fpnew_pkg::fpu_implementation_t FPUImplementation = '0,
    // derived parameters
    parameter int unsigned  InitialOffset = ((2**BankGroupAddrWidth) / 4) - 8
) (
    input  logic                        clk_i,
    input  logic                        rst_i,
    input  logic [31:0]                 hart_id_i,

    // Accelerator Interface - Slave
    input  acc_req_t                    acc_req_i,
    input  logic                        acc_req_valid_i,
    output logic                        acc_req_ready_o,
    output acc_resp_t                   acc_rsp_o,
    output logic                        acc_rsp_valid_o,
    input  logic                        acc_rsp_ready_i,

    // TCDM Data Interface for regular FP load/stores.
    output tcdm_req_t                   tcdm_req_o,
    input  tcdm_rsp_t                   tcdm_rsp_i,

    // TCDM Data Interface for snitch core and other FPU units
    input  tcdm_req_t                   tcdm_req_i,
    output tcdm_rsp_t                   tcdm_rsp_o,

    // Memory Interface for accelerated FP load/stores.
    output bank_req_t [3:0]             bank_req_o,
    input  bank_rsp_t [3:0]             bank_rsp_i,

    // FPU **un-timed** Side-channel
    input  fpnew_pkg::roundmode_e       fpu_rnd_mode_i,
    input  fpnew_pkg::fmt_mode_t        fpu_fmt_mode_i,
    output fpnew_pkg::status_t          fpu_status_o,

    // Configuration registers
    input  fpu_cfg_word_e               cfg_word_i,
    input  logic [3:0]                  cfg_bank_sel_i,
    output logic [31:0]                 cfg_rdata_o,
    input  logic [31:0]                 cfg_wdata_i,
    input  logic                        cfg_write_i,
    output logic                        cfg_ready_o,

    output fpu_trace_port_t             trace_port_o,
    output fpu_sequencer_trace_port_t   sequencer_tracer_port_o,
    output core_events_t                core_events_o
);
    // ---------------------
    // Protocol Typedefs
    // ---------------------
    typedef logic [BankGroupAddrWidth-1:0]  addr_t;
    typedef logic [DataWidth-1:0]           data_t;
    typedef logic [DataWidth/8-1:0]         strb_t;
    `TCDM_TYPEDEF_ALL(vfpr, addr_t, data_t, strb_t, logic)

    // ---------------------
    // WR phase signals
    // ---------------------
    vfpr_req_t wr_vfpr_req;
    vfpr_rsp_t wr_vfpr_rsp;
    logic [ScoreboardDepth-1:0] wr_sb_index;
    logic                       wr_sb_index_valid;            
    logic                       wr_acc_ready;

    // ---------------------
    // Offset configuration registers
    // ---------------------
    logic [3:0][BankGroupAddrWidth/4:0] cfg_offset_d, cfg_offset_q;
    logic [3:0][BankGroupAddrWidth/4:0] cfg_stride_d, cfg_stride_q;

    // ---------------------
    // Sequencer and configuration
    // ---------------------
    acc_req_t       seq_out, seq_out_q;
    logic           seq_out_valid, seq_out_valid_q;
    logic           seq_out_ready, seq_out_ready_q;

    logic seq_stride_inc;
    stitch_sequencer #(
        .AddrWidth  (AddrWidth),
        .DataWidth  (DataWidth),
        .Depth      (NumFPUSequencerInstr)
    ) i_stitch_fpu_sequencer ( 
        .clk_i,
        .rst_i,
        .fpu_id_i,
        // pragma translate_off
        .trace_port_o     (sequencer_tracer_port_o),
        // pragma translate_on
        .inp_qaddr_i      (acc_req_i.addr),
        .inp_qid_i        (acc_req_i.id),
        .inp_qdata_op_i   (acc_req_i.data_op),
        .inp_qdata_arga_i (acc_req_i.data_arga),
        .inp_qdata_argb_i (acc_req_i.data_argb),
        .inp_qdata_argc_i (acc_req_i.data_argc),
        .inp_qvalid_i     (acc_req_valid_i),
        .inp_qready_o     (acc_req_ready_o),
        .oup_qaddr_o      (seq_out.addr),
        .oup_qid_o        (seq_out.id),
        .oup_qdata_op_o   (seq_out.data_op),
        .oup_qdata_arga_o (seq_out.data_arga),
        .oup_qdata_argb_o (seq_out.data_argb),
        .oup_qdata_argc_o (seq_out.data_argc),
        .oup_qvalid_o     (seq_out_valid),
        .oup_qready_i     (seq_out_ready),
        .oup_inc_offset_o (seq_stride_inc) 
    );

    spill_register  #(
        .T          (acc_req_t),
        .Bypass     (!RegisterSequencer)
    ) i_spill_register_acc (
        .clk_i   ,
        .rst_ni     (~rst_i),
        .valid_i    (seq_out_valid),
        .ready_o    (seq_out_ready),
        .data_i     (seq_out),
        .valid_o    (seq_out_valid_q),
        .ready_i    (seq_out_ready_q),
        .data_o     (seq_out_q)
    );

    `FFAR(cfg_offset_q, cfg_offset_d, InitialOffset, clk_i, rst_i)
    `FFAR(cfg_stride_q, cfg_stride_d, '0, clk_i, rst_i)
 
    // write configuration registers
    logic stride;
    assign stride = seq_stride_inc & seq_out_valid & seq_out_ready;
    assign cfg_ready_o = seq_out_valid & seq_out_ready;
    for (genvar i = 0; i < 4; i++) begin
        always_comb begin
            cfg_stride_d[i] = cfg_offset_q[i] + (stride ? cfg_stride_q[i] : 0);
            cfg_stride_d[i] = cfg_stride_q[i];
            if (cfg_write_i & cfg_bank_sel_i[i] & cfg_ready_o) begin
                unique case (cfg_word_i)
                    fpu_cfg_word_e::Offset: begin
                        cfg_offset_d[i] = cfg_wdata_i;
                    end
                    fpu_cfg_word_e::Stride: begin
                        cfg_stride_d[i] = cfg_wdata_i;
                    end
                endcase
            end
        end
    end

    // read configuration registers
    always_comb begin
        unique case (cfg_word_i)
            fpu_cfg_word_e::Offset: begin
                cfg_rdata_o = cfg_offset_d[cfg_bank_sel_i[1:0]];
            end
            fpu_cfg_word_e::Stride: begin
                cfg_rdata_o = cfg_stride_d[cfg_bank_sel_i[1:0]];
            end
        endcase
    end

    // ---------------------
    // Scoreboard & Decoder
    // ---------------------
    typedef struct packed {
        logic [2:0][BankGroupAddrWidth-1:0] fpr_src_addrs;
        logic [4:0]                         rd_name;
        logic [BankGroupAddrWidth-1:0]      fpr_rd_addr;
        logic [ScoreboardDepth-1:0]         sb_rd_index;
        logic [2:0][4:0]                    fpr_srcs_en;
        fpnew_pkg::operation_e              fpu_op;
        fpnew_pkg::roundmode_e              rnd_mode;
        fpnew_pkg::fp_format_e              src_fmt; 
        fpnew_pkg::fp_format_e              dst_fmt;
        fpnew_pkg::int_format_e             int_fmt;
        fpu_operand_src_e [2:0]             operand_src;
        logic                               vectoral_op;
        logic                               is_load_op;
        logic                               is_store_op;
        logic                               is_fpu_op;
        logic                               is_csr_ins;
        logic                               rd_is_fpr;
        logic                               rd_is_acc;
        logic                               op_mode;
        fpu_ls_size_e                       ls_size;
        logic [AddrWidth-1:0]               lsu_addr;
        logic [31:0]                        acc_value;
    } sb_tag_t;

    sb_tag_t        sb_dec_tag, sb_dec_tag_out;
    logic           sb_dec_tag_valid, sb_dec_out_valid;
    logic           sb_dec_tag_ready, sb_dec_out_ready;

    logic [2:0][4:0]    fpr_srcs;
    logic [4:0]         fpr_rd;

    logic sb_full;
    logic [3:0] sb_collision;

    assign seq_out_ready_q = ~sb_full & sb_dec_tag_ready;
    assign sb_dec_tag_valid = ~sb_full & seq_out_valid_q;

    stitch_decoder i_decode (
        .ins_i          (seq_out_q.data_op),
        .fpu_rnd_mode_i (fpu_rnd_mode_i),
        .fpu_fmt_mode_i (fpu_fmt_mode_i),
        .error_o        (acc_rsp_o.error),
        .fpu_op_o       (sb_dec_tag.fpu_op),
        .rnd_mode_o     (sb_dec_tag.rnd_mode),
        .src_fmt_o      (sb_dec_tag.src_fmt),
        .dst_fmt_o      (sb_dec_tag.dst_fmt),
        .int_fmt_o      (sb_dec_tag.int_fmt),
        .fpr_srcs_o     (fpr_srcs),
        .fpr_srcs_en_o  (sb_dec_tag.fpr_srcs_en),
        .fpr_rd_o       (fpr_rd),
        .operand_src_o  (sb_dec_tag.operand_src),
        .vectoral_op_o  (sb_dec_tag.vectoral_op),
        .is_load_op_o   (sb_dec_tag.is_load_op),
        .is_store_op_o  (sb_dec_tag.is_store_op),
        .is_fpu_op_o    (sb_dec_tag.is_fpu_op),
        .is_csr_ins_o   (sb_dec_tag.is_csr_ins),
        .rd_is_fpr_o    (sb_dec_tag.rd_is_fpr),
        .rd_is_acc_o    (sb_dec_tag.rd_is_acc),
        .op_mode_o      (sb_dec_tag.op_mode),
        .ls_size_o      (sb_dec_tag.ls_size)
    );

    assign sb_dec_tag.rd_name = fpr_rd;
    assign sb_dec_tag.lsu_addr = seq_out.data_argc[AddrWidth-1:0];
    assign sb_dec_tag.acc_value = seq_out.data_arga[31:0];

    for (genvar i = 0; i < 3; i++) begin
        always_comb begin
            case (fpr_srcs[i][4:3])
                0: sb_dec_tag.fpr_src_addrs[i] = {cfg_offset_q[0], fpr_srcs[i][2:0]};
                1: sb_dec_tag.fpr_src_addrs[i] = {cfg_offset_q[1], fpr_srcs[i][2:0]};
                2: sb_dec_tag.fpr_src_addrs[i] = {cfg_offset_q[2], fpr_srcs[i][2:0]};
                3: sb_dec_tag.fpr_src_addrs[i] = {cfg_offset_q[3], fpr_srcs[i][2:0]};
                default:;
            endcase
        end
    end

    always_comb begin
        case (fpr_rd[4:3])
            0: sb_dec_tag.fpr_rd_addr = {cfg_offset_q[0], fpr_rd[2:0]};
            1: sb_dec_tag.fpr_rd_addr = {cfg_offset_q[1], fpr_rd[2:0]};
            2: sb_dec_tag.fpr_rd_addr = {cfg_offset_q[2], fpr_rd[2:0]};
            3: sb_dec_tag.fpr_rd_addr = {cfg_offset_q[3], fpr_rd[2:0]};
            default:;
        endcase
    end

    stitch_sb #(
        .AddrWidth      (BankGroupAddrWidth),
        .Depth          (ScoreboardDepth),
        .NumTestAddrs   (4)
    ) i_sb (
        .clk_i,
        .rst_i,
        .push_rd_addr_i         (sb_dec_tag.fpr_rd_addr),
        .push_valid_i           (sb_dec_tag_valid & seq_out_ready_q & sb_dec_tag.rd_is_fpr),
        .entry_index_o          (sb_dec_tag.sb_rd_index),
        .pop_index_i            (wr_sb_index),
        .pop_valid_i            (wr_sb_index_valid),
        .test_rd_addr_i         ({
            fpr_src_addrs[0], 
            fpr_src_addrs[1], 
            fpr_src_addrs[2], 
            sb_dec_tag.fpr_rd_addr
        }),
        .test_addr_present_o    (sb_collision),
        .full_o                 (sb_full)
    );

    spill_register  #(
        .T          (sb_dec_out_t),
        .Bypass     (!RegisterDecoder)
    ) i_spill_register_acc (
        .clk_i,
        .rst_ni     (~rst_i),
        .valid_i    (sb_dec_tag_valid),
        .ready_o    (sb_dec_tag_ready),
        .data_i     (sb_dec_tag),
        .valid_o    (sb_dec_out_valid),
        .ready_i    (sb_dec_out_ready),
        .data_o     (sb_dec_out)
    );

    // ---------------------
    // Virtual Register File
    // ---------------------
    typedef struct packed {
        logic [4:0]                         rd_name;
        logic [BankGroupAddrWidth-1:0]      fpr_rd_addr;
        logic [ScoreboardDepth-1:0]         sb_rd_index;
        fpnew_pkg::operation_e              fpu_op;
        fpnew_pkg::roundmode_e              rnd_mode;
        fpnew_pkg::fp_format_e              src_fmt; 
        fpnew_pkg::fp_format_e              dst_fmt;
        fpnew_pkg::int_format_e             int_fmt;
        fpu_operand_src_e [2:0]             operand_src;
        logic                               vectoral_op;
        logic                               is_load_op;
        logic                               is_store_op;
        logic                               is_fpu_op;
        logic                               is_csr_ins;
        logic                               rd_is_fpr;
        logic                               rd_is_acc;
        logic                               op_mode;
        fpu_ls_size_e                       ls_size;
        logic [AddrWidth-1:0]               lsu_addr;
    } vfpr_tag_t;

    vfpr_tag_t      vfpr_tag_in, vfpr_tag_out;
    logic           vfpr_ready, vfpr_out_valid;
                                ;

    logic [2:0][FLEN-1:0] vfpr_operands;
    logic [2:0][FLEN-1:0] fpu_operands;

    // repack struct
    assign vfpr_tag_in.rd_name = sb_dec_out.rd_name;
    assign vfpr_tag_in.fpr_rd_addr = sb_dec_out.rd_name;
    assign vfpr_tag_in.sb_rd_index = sb_dec_out.sb_rd_index;
    assign vfpr_tag_in.fpu_op = sb_dec_out.fpu_op;
    assign vfpr_tag_in.rnd_mode = sb_dec_out.rnd_mode;
    assign vfpr_tag_in.src_fmt = sb_dec_out.src_fmt;
    assign vfpr_tag_in.dst_fmt = sb_dec_out.dst_fmt;
    assign vfpr_tag_in.int_fmt = sb_dec_out.int_fmt;
    assign vfpr_tag_in.operand_src = sb_dec_out.operand_src;
    assign vfpr_tag_in.vectoral_op = sb_dec_out.vectoral_op;
    assign vfpr_tag_in.is_load_op = sb_dec_out.is_load_op;
    assign vfpr_tag_in.is_store_op = sb_dec_out.is_store_op;
    assign vfpr_tag_in.is_fpu_op = sb_dec_out.is_fpu_op;
    assign vfpr_tag_in.is_csr_ins = sb_dec_out.is_csr_ins;
    assign vfpr_tag_in.rd_is_fpr = sb_dec_out.rd_is_fpr;
    assign vfpr_tag_in.rd_is_acc = sb_dec_out.rd_is_acc;
    assign vfpr_tag_in.op_mode = sb_dec_out.op_mode;
    assign vfpr_tag_in.ls_size = sb_dec_out.ls_size;
    assign vfpr_tag_in.lsu_addr = sb_dec_out.lsu_addr;

    stitch_vfpr #(
        .DataWidth(DataWidth),
        .AddrWidth(TotalAddrWidth),
        .tag_t(vfpr_out_t),
        .vfpr_req_t(vfpr_req_t),
        .vfpr_rsp_t(vfpr_rsp_t),
        .dbankreq_t(bank_req_t),
        .dbankrsp_t(bank_rsp_t)
    ) i_vfpr (
        .clk_i,
        .rst_i,
        .raddr_i        (sb_dec_out_q),
        .rtag_i         (vfpr_tag_in),
        .ren_i          (sb_dec_out_q.fpr_srcs_en),
        .rvalid_i       (sb_dec_out_valid),
        .rready_o       (sb_dec_out_ready),
        .wr_port_req_i  (wr_vfpr_req),
        .wr_port_rsp_o  (wr_vfpr_rsp),
        .rdata_o        (vfpr_operands),
        .rtag_o         (vfpr_tag_out),
        .rdata_valid_o  (vfpr_out_valid),
        .rdata_ready_i  (vfpr_out_ready),
        .mem_req_o      (bank_req_o),
        .mem_rsp_i      (bank_rsp_i)
    );

    for (genvar i = 0; i < 3; i++) begin: gen_operand_select
        always_comb begin
            unique case (vfpr_tag_out.operand_src[i])
                None: begin
                    fpu_operands[i] = '1;
                end
                AccBus: begin
                    fpu_operands[i] = acc_qdata[i];
                end
                RegA, RegB, RegBRep, RegC, RegDest: begin
                    fpu_operands[i] = vfpr_operands[i];
                    // Replicate if needed
                    if (vfpr_tag_out.operand_src[i] == RegBRep) begin
                        unique case (src_fmt)
                        fpnew_pkg::FP32:    fpu_operands[i] = {(FLEN / 32){fpu_operands[i][31:0]}};
                        fpnew_pkg::FP16,
                        fpnew_pkg::FP16ALT: fpu_operands[i] = {(FLEN / 16){fpu_operands[i][15:0]}};
                        fpnew_pkg::FP8,
                        fpnew_pkg::FP8ALT:  fpu_operands[i] = {(FLEN /  8){fpu_operands[i][ 7:0]}};
                        default:            fpu_operands[i] = fpu_operands[i][FLEN-1:0];
                        endcase
                    end
                end
                default: begin
                    fpu_operands[i] = '0;
                end
            endcase
        end
    end

    // ---------------------
    // FPU
    // ---------------------
    typedef struct packed {
        logic [4:0]                         rd_name;
        logic [2:0][BankGroupAddrWidth-1:0] fpr_rd_addr;
        logic [2:0][ScoreboardDepth-1:0]    sb_rd_index;
        logic                               rd_is_acc;
        logic                               rd_is_fpr;
    } fpu_tag_t;

    data_t          fpu_result;
    fpu_tag_t       fpu_tag_in, fpu_tag_out;
    logic           fpu_in_valid, fpu_out_valid;
    logic           fpu_in_ready, fpu_out_ready;

    assign fpu_in_valid = vfpr_out_valid & vfpr_tag_out.is_fpu_op;

    assign fpu_tag_in.rd_name = vfpr_tag_out.rd_name;
    assign fpu_tag_in.fpr_rd_addr = vfpr_tag_out.fpr_rd_addr;
    assign fpu_tag_in.sb_rd_index = vfpr_tag_out.sb_rd_index;
    assign fpu_tag_in.rd_is_acc = vfpr_tag_out.rd_is_acc;
    assign fpu_tag_in.rd_is_fpr = vfpr_tag_out.rd_is_fpr;

    stitch_fpu #(
        .RVF        (RVF),
        .RVD        (RVD),
        .XF16       (XF16),
        .XF16ALT    (XF16ALT),
        .XF8        (XF8),
        .XF8ALT     (XF8ALT),
        .XFVEC      (XFVEC),
        .FLEN       (FLEN),
        .FPUImplementation  (FPUImplementation),
        .RegisterFPUIn      (RegisterFPUIn),
        .RegisterFPUOut     (RegisterFPUOut)
    ) i_fpu (
        .clk_i,
        .rst_ni         (~rst_i),
        .hart_id_i      (hart_id_i),
        .operands_i     (fpu_operands),
        .rnd_mode_i     (vfpr_tag_out.rnd_mode),
        .op_i           (vfpr_tag_out.fpu_op),
        .op_mod_i       (vfpr_tag_out.op_mode), // Sign of operand?
        .src_fmt_i      (vfpr_tag_out.src_fmt),
        .dst_fmt_i      (vfpr_tag_out.dst_fmt),
        .int_fmt_i      (vfpr_tag_out.int_fmt),
        .vectorial_op_i (vfpr_tag_out.vectoral_op),
        .tag_i          (fpu_tag_in),
        .in_valid_i     (fpu_in_valid),
        .in_ready_o     (fpu_in_ready),
        .result_o       (fpu_result),
        .status_o       (fpu_status_o),
        .tag_o          (fpu_tag_out),
        .out_valid_o    (fpu_out_valid),
        .out_ready_i    (fpu_out_ready)
    );

    // ---------------------
    // LSU
    // ---------------------
    typedef struct packed {
        logic [2:0][BankGroupAddrWidth-1:0] fpr_rd_addr;
        logic [2:0][ScoreboardDepth-1:0]    sb_rd_index;
    } lsu_tag_t;

    data_t          lsu_result;
    lsu_tag_t       lsu_tag_in, lsu_tag_out;
    logic           lsu_in_valid, lsu_out_valid;
    logic           lsu_in_ready, lsu_out_ready;

    assign lsu_in_valid = vfpr_out_valid & (vfpr_tag_out.is_store_op | vfpr_tag_out.is_load_op);

    snitch_lsu #(
        .AddrWidth              (AddrWidth),
        .DataWidth              (DataWidth),
        .dreq_t                 (dreq_t),
        .drsp_t                 (drsp_t),
        .tag_t                  (lsu_tag_t),
        .NumOutstandingMem      (NumFPOutstandingMem),
        .NumOutstandingLoads    (NumFPOutstandingLoads),
        .NaNBox                 (1'b1)
    ) i_snitch_lsu (
        .clk_i,
        .rst_i,
        .lsu_qtag_i     (lsu_tag_in),
        .lsu_qwrite_i   (vfpr_tag_out.is_store_op),
        .lsu_qsigned_i  (1'b1), // all floating point loads are signed
        .lsu_qaddr_i    (vfpr_tag_out.lsu_addr),
        .lsu_qdata_i    (fpu_operands[1]),
        .lsu_qsize_i    (vfpr_tag_out.ls_size),
        .lsu_qamo_i     (reqrsp_pkg::AMONone),
        .lsu_qvalid_i   (lsu_in_valid),
        .lsu_qready_o   (lsu_in_ready),
        .lsu_pdata_o    (lsu_result),
        .lsu_ptag_o     (lsu_tag_out),
        .lsu_perror_o   (), // ignored for the moment
        .lsu_pvalid_o   (lsu_out_valid),
        .lsu_pready_i   (lsu_out_ready),
        .lsu_empty_o    (/* unused */),
        .data_req_o     (tcdm_req_o),
        .data_rsp_i     (tcdm_rsp_i)
    );

    assign vfpr_out_ready = (lsu_in_valid & lsu_in_ready) 
                          | (fpu_in_valid & fpu_in_ready);

    // ---------------------
    // WR
    // ---------------------

    // determine who gets to access the register file
    typedef enum logic [1:0] {
        FPU,
        LSU,
        TCDM,
        None
    } wr_sel_e;

    logic [DataWidth-1:0] nan_boxed_arga;
    assign nan_boxed_arga = {{(DataWidth-32){1'b1}}, sb_dec_out.acc_value};

    wr_sel_e wr_state_q, wr_state_d;
    wr_sel_e wr_sel;

    `FFAR(wr_state_q, wr_state_d, '0, clk_i, rst_i)

    always_comb begin
        lsu_out_ready = 0;
        fpu_out_ready = 0;
        wr_acc_ready = 0;
        wr_state_d = wr_state_q;
        wr_sb_index_valid = 0;
        tcdm_rsp_o.q_ready = 0;

        if (wr_vfpr_rsp.p_valid) begin
            wr_state_d = None;
        end

        // if we are in the none state then we are free
        // to transition to a new state. Otherwise wait to avoid
        // violating amba'ness
        if (fpu_out_valid & wr_state_d == None) begin
            wr_sel = FPU;
        end else if (lsu_out_valid & wr_state_d == None) begin
            wr_sel = LSU;
        end else if (lsu_out_valid & wr_state_d == None) begin
            wr_sel = TCDM;
        end else begin
            wr_sel = None;
        end

        unique case (wr_sel)
            FPU: begin
                wr_state_d = FPU;
                wr_vfpr_req.q.addr = fpu_tag_out.fpr_rd_addr;
                wr_vfpr_req.q.data = fpu_result;
                wr_vfpr_req.q.write = '1;
                wr_vfpr_req.q.strb = '1;
                wr_vfpr_req.q.amo = reqrsp_pkg::AMONone;
                wr_vfpr_req.q.user = '0;

                wr_vfpr_req.q_valid = fpu_out_valid;
                fpu_out_ready = (wr_vfpr_rsp.q_ready & fpu_tag_out.rd_is_fpr) 
                              | (acc_rsp_ready_i & fpu_tag_out.rd_is_acc);
                wr_sb_index = fpu_tag_out.sb_rd_index;
                wr_sb_index_valid = fpu_out_valid & fpu_out_ready;
            end
            LSU: begin
                wr_state_d = LSU;
                wr_vfpr_req.q.addr = lsu_tag_out.fpr_rd_addr;
                wr_vfpr_req.q.data = lsu_result;
                wr_vfpr_req.q.write = '1;
                wr_vfpr_req.q.strb = '1;
                wr_vfpr_req.q.amo = reqrsp_pkg::AMONone;
                wr_vfpr_req.q.user = '0;

                wr_vfpr_req.q_valid = lsu_out_valid;
                lsu_out_ready = wr_vfpr_rsp.q_ready;
                wr_sb_index = lsu_tag_out.sb_rd_index;
                wr_sb_index_valid = lsu_out_valid & lsu_out_ready;
            end
            TCDM: begin
                wr_state_d = TCDM;
                wr_sb_index = 'x;
                wr_sb_index_valid = 0;
                wr_vfpr_req = tcdm_req_i;
                tcdm_rsp_o.q_ready = wr_vfpr_rsp.q_ready;
            end
            default:;
        endcase
    end

    // if the tcdm requested something, make sure we route the response appropriately
    // the other datapaths don't care about responses
    assign tcdm_rsp_o.p = wr_vfpr_rsp.p;
    assign tcdm_rsp_o.p_valid = wr_vfpr_rsp.p_valid & (wr_state_q == TCDM);

    // determine if fpu result is written to accelerator bus
    assign acc_resp_o.id = fpu_tag_out.rd_name;
    assign acc_resp_o.data = fpu_result;
    assign acc_rsp_valid_o = (fpu_tag_out.rd_is_acc & fpu_out_valid);

    assign trace_port_o.source       = snitch_pkg::SrcFpu;
    assign trace_port_o.acc_q_hs     = (acc_req_valid_q  && acc_req_ready_q );
    assign trace_port_o.fpu_out_hs   = (fpu_out_valid && fpu_out_ready );
    assign trace_port_o.lsu_q_hs     = (lsu_qvalid    && lsu_qready    );
    assign trace_port_o.op_in        = acc_req_q.data_op;
    assign trace_port_o.rs1          = rs1;
    assign trace_port_o.rs2          = rs2;
    assign trace_port_o.rs3          = rs3;
    assign trace_port_o.rd           = rd;
    assign trace_port_o.op_sel_0     = op_select[0];
    assign trace_port_o.op_sel_1     = op_select[1];
    assign trace_port_o.op_sel_2     = op_select[2];
    assign trace_port_o.src_fmt      = src_fmt;
    assign trace_port_o.dst_fmt      = dst_fmt;
    assign trace_port_o.int_fmt      = int_fmt;
    assign trace_port_o.acc_qdata_0  = acc_qdata[0];
    assign trace_port_o.acc_qdata_1  = acc_qdata[1];
    assign trace_port_o.acc_qdata_2  = acc_qdata[2];
    assign trace_port_o.op_0         = op[0];
    assign trace_port_o.op_1         = op[1];
    assign trace_port_o.op_2         = op[2];
    assign trace_port_o.use_fpu      = use_fpu;
    assign trace_port_o.fpu_in_rd    = fpu_tag_in.rd;
    assign trace_port_o.fpu_in_acc   = fpu_tag_in.acc;
    assign trace_port_o.ls_size      = ls_size;
    assign trace_port_o.is_load      = is_load;
    assign trace_port_o.is_store     = is_store;
    assign trace_port_o.lsu_qaddr    = i_snitch_lsu.lsu_qaddr_i;
    assign trace_port_o.lsu_rd       = lsu_rd;
    assign trace_port_o.acc_wb_ready = (result_select == ResAccBus);
    assign trace_port_o.fpu_out_acc  = fpu_tag_out.acc;
    assign trace_port_o.fpr_waddr    = fpr_waddr[0];
    assign trace_port_o.fpr_wdata    = fpr_wdata[0];
    assign trace_port_o.fpr_we       = fpr_we[0];
endmodule

