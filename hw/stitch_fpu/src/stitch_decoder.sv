module stitch_decoder import snitch_pkg::*; #(
    parameter int unsigned InsWidth = 32
) (
    input  logic [InsWidth-1:0]     ins_i,
    input  fpnew_pkg::roundmode_e   fpu_rnd_mode_i,
    input  fpnew_pkg::fmt_mode_t    fpu_fmt_mode_i,

    output logic                    error_o,
    output fpnew_pkg::operation_e   fpu_op_o,
    output fpnew_pkg::roundmode_e   rnd_mode_o,
    output fpnew_pkg::fp_format_e   src_fmt_o, 
    output fpnew_pkg::fp_format_e   dst_fmt_o,
    output fpnew_pkg::int_format_e  int_fmt_o,
    output logic [2:0][4:0]         fpr_srcs_o,
    output logic [2:0]              fpr_srcs_en_o,
    output logic [4:0]              fpr_rd_o,
    output fpu_operand_src_e [2:0]  operand_src_o,
    output logic                    vectoral_op_o,
    output logic                    is_load_op_o,
    output logic                    is_store_op_o,
    output logic                    is_fpu_op_o,
    output logic                    is_csr_ins_o,
    output logic                    rd_is_fpr_o,
    output logic                    rd_is_acc_o,
    output logic                    op_mode_o,
    output fpu_ls_size_e            ls_size_o
);

    logic set_dyn_rm;
    logic [4:0] rs1, rs2, rs3, rd;
    always_comb begin
        error_o = 1'b0;
        fpu_op_o = fpnew_pkg::ADD;
        is_fpu_op_o = 1'b1;
        fpu_rnd_mode_o = (fpnew_pkg::roundmode_e'(ins_i[14:12]) == fpnew_pkg::DYN)
                    ? fpu_rnd_mode_i
                    : fpnew_pkg::roundmode_e'(ins_i[14:12]);

        set_dyn_rm = 1'b0;

        src_fmt_o = fpnew_pkg::FP32;
        dst_fmt_o = fpnew_pkg::FP32;
        int_fmt_o = fpnew_pkg::INT32;

        operand_src_o[0] = None;
        operand_src_o[1] = None;
        operand_src_o[2] = None;

        fpr_rd_o = ins_i[11:7]; // rd
        fpr_srcs_o[0] = ins_i[19:15]; // rs1
        fpr_srcs_o[1] = ins_i[24:20]; // rs2
        fpr_srcs_o[2] = ins_i[31:27]; // rs3

        vectoral_op_o = 1'b0;
        op_mode_o = 1'b0;

        rd_is_acc_o = 1'b0; // RD is on accelerator bus

        is_store_op_o = 1'b0;
        is_load_op_o = 1'b0;
        ls_size_o = Word;

        rd_is_fpr_o = 1'b1; // dst is floating point register
        is_csr_ins_o = 1'b0; // is a csr instruction
        
        unique casez (ins_i)
            // FP - FP Operations
            // Single Precision
            riscv_instr::FADD_S: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
            end
            riscv_instr::FSUB_S: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                op_mode_o = 1'b1;
            end
            riscv_instr::FMUL_S: begin
                fpu_op_o = fpnew_pkg::MUL;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
            end
            riscv_instr::FDIV_S: begin  // currently illegal
                fpu_op_o = fpnew_pkg::DIV;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
            end
            riscv_instr::FSGNJ_S,
            riscv_instr::FSGNJN_S,
            riscv_instr::FSGNJX_S: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
            end
            riscv_instr::FMIN_S,
            riscv_instr::FMAX_S: begin
                fpu_op_o = fpnew_pkg::MINMAX;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
            end
            riscv_instr::FSQRT_S: begin  // currently illegal
                fpu_op_o = fpnew_pkg::SQRT;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegA;
            end
            riscv_instr::FMADD_S: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
            end
            riscv_instr::FMSUB_S: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                op_mode_o      = 1'b1;
            end
            riscv_instr::FNMSUB_S: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
            end
            riscv_instr::FNMADD_S: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                op_mode_o      = 1'b1;
            end
            // Vectorial Single Precision
            riscv_instr::VFADD_S,
            riscv_instr::VFADD_R_S: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFADD_R_S}) operand_src_o[2] = RegBRep;
            end
            riscv_instr::VFSUB_S,
            riscv_instr::VFSUB_R_S: begin
                fpu_op_o  = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                op_mode_o      = 1'b1;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFSUB_R_S}) operand_src_o[2] = RegBRep;
            end
            riscv_instr::VFMUL_S,
            riscv_instr::VFMUL_R_S: begin
                fpu_op_o = fpnew_pkg::MUL;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFMUL_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFDIV_S,
            riscv_instr::VFDIV_R_S: begin  // currently illegal
                fpu_op_o = fpnew_pkg::DIV;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFDIV_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFMIN_S,
            riscv_instr::VFMIN_R_S: begin
                fpu_op_o = fpnew_pkg::MINMAX;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RNE;
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFMIN_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFMAX_S,
            riscv_instr::VFMAX_R_S: begin
                fpu_op_o = fpnew_pkg::MINMAX;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RTZ;
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFMAX_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSQRT_S: begin // currently illegal
                fpu_op_o = fpnew_pkg::SQRT;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegA;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
            end
            riscv_instr::VFMAC_S,
            riscv_instr::VFMAC_R_S: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFMAC_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFMRE_S,
            riscv_instr::VFMRE_R_S: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFMRE_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSGNJ_S,
            riscv_instr::VFSGNJ_R_S: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RNE;
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFSGNJ_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSGNJN_S,
            riscv_instr::VFSGNJN_R_S: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RTZ;
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFSGNJN_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSGNJX_S,
            riscv_instr::VFSGNJX_R_S: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RDN;
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFSGNJX_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSUM_S,
            riscv_instr::VFNSUM_S: begin
                fpu_op_o = fpnew_pkg::VSUM;
                operand_src_o[0] = RegA;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP32;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFNSUM_S}) op_mode_o = 1'b1;
            end
            riscv_instr::VFCPKA_S_S: begin
                fpu_op_o = fpnew_pkg::CPKAB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
            end
            riscv_instr::VFCPKA_S_D: begin
                fpu_op_o = fpnew_pkg::CPKAB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP64;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
            end
            // Double Precision
            riscv_instr::FADD_D: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FSUB_D: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                op_mode_o = 1'b1;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FMUL_D: begin
                fpu_op_o = fpnew_pkg::MUL;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FDIV_D: begin
                fpu_op_o = fpnew_pkg::DIV;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FSGNJ_D,
            riscv_instr::FSGNJN_D,
            riscv_instr::FSGNJX_D: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FMIN_D,
            riscv_instr::FMAX_D: begin
                fpu_op_o = fpnew_pkg::MINMAX;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FSQRT_D: begin
                fpu_op_o = fpnew_pkg::SQRT;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegA;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FMADD_D: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FMSUB_D: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FNMSUB_D: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FNMADD_D: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FCVT_S_D: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP32;
            end
            riscv_instr::FCVT_D_S: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            // [Alternate] Half Precision
            riscv_instr::FADD_H: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FSUB_H: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                op_mode_o = 1'b1;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FMUL_H: begin
                fpu_op_o = fpnew_pkg::MUL;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FDIV_H: begin
                fpu_op_o = fpnew_pkg::DIV;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FSGNJ_H,
            riscv_instr::FSGNJN_H,
            riscv_instr::FSGNJX_H: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FMIN_H,
            riscv_instr::FMAX_H: begin
                fpu_op_o = fpnew_pkg::MINMAX;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FSQRT_H: begin
                fpu_op_o = fpnew_pkg::SQRT;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegA;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FMADD_H: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FMSUB_H: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FNMSUB_H: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FNMADD_H: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::VFSUM_H,
            riscv_instr::VFNSUM_H: begin
                fpu_op_o = fpnew_pkg::VSUM;
                operand_src_o[0] = RegA;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFNSUM_H}) op_mode_o = 1'b1;
            end
            riscv_instr::FMULEX_S_H: begin
                fpu_op_o = fpnew_pkg::MUL;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP32;
            end
            riscv_instr::FMACEX_S_H: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP32;
            end
            riscv_instr::FCVT_S_H: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP32;
            end
            riscv_instr::FCVT_H_S: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP16;
            end
            riscv_instr::FCVT_D_H: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FCVT_H_D: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP16;
            end
            riscv_instr::FCVT_H_H: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
            end
            // Vectorial [alternate] Half Precision
            riscv_instr::VFADD_H,
            riscv_instr::VFADD_R_H: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFADD_R_H}) operand_src_o[2] = RegBRep;
            end
            riscv_instr::VFSUB_H,
            riscv_instr::VFSUB_R_H: begin
                fpu_op_o  = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFSUB_R_H}) operand_src_o[2] = RegBRep;
            end
            riscv_instr::VFMUL_H,
            riscv_instr::VFMUL_R_H: begin
                fpu_op_o = fpnew_pkg::MUL;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFMUL_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFDIV_H,
            riscv_instr::VFDIV_R_H: begin
                fpu_op_o = fpnew_pkg::DIV;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFDIV_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFMIN_H,
            riscv_instr::VFMIN_R_H: begin
                fpu_op_o = fpnew_pkg::MINMAX;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RNE;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFMIN_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFMAX_H,
            riscv_instr::VFMAX_R_H: begin
                fpu_op_o = fpnew_pkg::MINMAX;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                fpu_rnd_mode_o = fpnew_pkg::RTZ;
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFMAX_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSQRT_H: begin
                fpu_op_o = fpnew_pkg::SQRT;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegA;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
            end
            riscv_instr::VFMAC_H,
            riscv_instr::VFMAC_R_H: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFMAC_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFMRE_H,
            riscv_instr::VFMRE_R_H: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFMRE_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSGNJ_H,
            riscv_instr::VFSGNJ_R_H: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RNE;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFSGNJ_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSGNJN_H,
            riscv_instr::VFSGNJN_R_H: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RTZ;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFSGNJN_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSGNJX_H,
            riscv_instr::VFSGNJX_R_H: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RDN;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFSGNJX_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFCPKA_H_S: begin
                fpu_op_o = fpnew_pkg::CPKAB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
            end
            riscv_instr::VFCPKB_H_S: begin
                fpu_op_o = fpnew_pkg::CPKAB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
            end
            riscv_instr::VFCVT_S_H,
            riscv_instr::VFCVTU_S_H: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP32;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCVTU_S_H}) op_mode_o = 1'b1;
            end
            riscv_instr::VFCVT_H_S,
            riscv_instr::VFCVTU_H_S: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCVTU_H_S}) op_mode_o = 1'b1;
            end
            riscv_instr::VFCPKA_H_D: begin
                fpu_op_o = fpnew_pkg::CPKAB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
            end
            riscv_instr::VFCPKB_H_D: begin
                fpu_op_o = fpnew_pkg::CPKAB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
            end
            riscv_instr::VFDOTPEX_S_H,
            riscv_instr::VFDOTPEX_S_R_H: begin
                fpu_op_o = fpnew_pkg::SDOTP;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP32;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFDOTPEX_S_R_H}) operand_src_o[2] = RegBRep;
            end
            riscv_instr::VFNDOTPEX_S_H,
            riscv_instr::VFNDOTPEX_S_R_H: begin
                fpu_op_o = fpnew_pkg::SDOTP;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP32;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFNDOTPEX_S_R_H}) operand_src_o[2] = RegBRep;
            end
            riscv_instr::VFSUMEX_S_H,
            riscv_instr::VFNSUMEX_S_H: begin
                fpu_op_o = fpnew_pkg::EXVSUM;
                operand_src_o[0] = RegA;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP32;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFNSUMEX_S_H}) op_mode_o = 1'b1;
            end
            // [Alternate] Quarter Precision
            riscv_instr::FADD_B: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FSUB_B: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                op_mode_o = 1'b1;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FMUL_B: begin
                fpu_op_o = fpnew_pkg::MUL;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FDIV_B: begin
                fpu_op_o = fpnew_pkg::DIV;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FSGNJ_B,
            riscv_instr::FSGNJN_B,
            riscv_instr::FSGNJX_B: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FMIN_B,
            riscv_instr::FMAX_B: begin
                fpu_op_o = fpnew_pkg::MINMAX;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FSQRT_B: begin
                fpu_op_o = fpnew_pkg::SQRT;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegA;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FMADD_B: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FMSUB_B: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FNMSUB_B: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FNMADD_B: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegC;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::VFSUM_B,
            riscv_instr::VFNSUM_B: begin
                fpu_op_o = fpnew_pkg::VSUM;
                operand_src_o[0] = RegA;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFNSUM_B}) op_mode_o = 1'b1;
            end
            riscv_instr::FMULEX_S_B: begin
                fpu_op_o = fpnew_pkg::MUL;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP32;
            end
            riscv_instr::FMACEX_S_B: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP32;
            end
            riscv_instr::FCVT_S_B: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP32;
            end
            riscv_instr::FCVT_B_S: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP8;
            end
            riscv_instr::FCVT_D_B: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP64;
            end
            riscv_instr::FCVT_B_D: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP8;
            end
            riscv_instr::FCVT_H_B: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP16;
            end
            riscv_instr::FCVT_B_H: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP8;
            end
            // Vectorial [alternate] Quarter Precision
            riscv_instr::VFADD_B,
            riscv_instr::VFADD_R_B: begin
                fpu_op_o = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFADD_R_B}) operand_src_o[2] = RegBRep;
            end
            riscv_instr::VFSUB_B,
            riscv_instr::VFSUB_R_B: begin
                fpu_op_o  = fpnew_pkg::ADD;
                operand_src_o[1] = RegA;
                operand_src_o[2] = RegB;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFSUB_R_B}) operand_src_o[2] = RegBRep;
            end
            riscv_instr::VFMUL_B,
            riscv_instr::VFMUL_R_B: begin
                fpu_op_o = fpnew_pkg::MUL;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFMUL_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFDIV_B,
            riscv_instr::VFDIV_R_B: begin
                fpu_op_o = fpnew_pkg::DIV;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFDIV_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFMIN_B,
            riscv_instr::VFMIN_R_B: begin
                fpu_op_o = fpnew_pkg::MINMAX;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RNE;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFMIN_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFMAX_B,
            riscv_instr::VFMAX_R_B: begin
                fpu_op_o = fpnew_pkg::MINMAX;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                fpu_rnd_mode_o = fpnew_pkg::RTZ;
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFMAX_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSQRT_B: begin
                fpu_op_o = fpnew_pkg::SQRT;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegA;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
            end
            riscv_instr::VFMAC_B,
            riscv_instr::VFMAC_R_B: begin
                fpu_op_o = fpnew_pkg::FMADD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFMAC_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFMRE_B,
            riscv_instr::VFMRE_R_B: begin
                fpu_op_o = fpnew_pkg::FNMSUB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFMRE_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSGNJ_B,
            riscv_instr::VFSGNJ_R_B: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RNE;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFSGNJ_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSGNJN_B,
            riscv_instr::VFSGNJN_R_B: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RTZ;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFSGNJN_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFSGNJX_B,
            riscv_instr::VFSGNJX_R_B: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                fpu_rnd_mode_o = fpnew_pkg::RDN;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o = 1'b1;
                if (ins_i inside {riscv_instr::VFSGNJX_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFCPKA_B_S,
            riscv_instr::VFCPKB_B_S: begin
                fpu_op_o = fpnew_pkg::CPKAB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP8;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCPKB_B_S}) op_mode_o = 1;
            end
            riscv_instr::VFCPKC_B_S,
            riscv_instr::VFCPKD_B_S: begin
                fpu_op_o = fpnew_pkg::CPKCD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP8;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCPKD_B_S}) op_mode_o = 1;
            end
            riscv_instr::VFCPKA_B_D,
            riscv_instr::VFCPKB_B_D: begin
                fpu_op_o = fpnew_pkg::CPKAB;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP8;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCPKB_B_D}) op_mode_o = 1;
            end
            riscv_instr::VFCPKC_B_D,
            riscv_instr::VFCPKD_B_D: begin
                fpu_op_o = fpnew_pkg::CPKCD;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP8;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCPKD_B_D}) op_mode_o = 1;
            end
            riscv_instr::VFCVT_S_B,
            riscv_instr::VFCVTU_S_B: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP32;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCVTU_S_B}) op_mode_o = 1'b1;
            end
            riscv_instr::VFCVT_B_S,
            riscv_instr::VFCVTU_B_S: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP8;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCVTU_B_S}) op_mode_o = 1'b1;
            end
            riscv_instr::VFCVT_H_H,
            riscv_instr::VFCVTU_H_H: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCVTU_H_H}) op_mode_o = 1'b1;
            end
            riscv_instr::VFCVT_H_B,
            riscv_instr::VFCVTU_H_B: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCVTU_H_B}) op_mode_o = 1'b1;
            end
            riscv_instr::VFCVT_B_H,
            riscv_instr::VFCVTU_B_H: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP8;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCVTU_B_H}) op_mode_o = 1'b1;
            end
            riscv_instr::VFCVT_B_B,
            riscv_instr::VFCVTU_B_B: begin
                fpu_op_o = fpnew_pkg::F2F;
                operand_src_o[0] = RegA;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCVTU_B_B}) op_mode_o = 1'b1;
            end
            riscv_instr::VFDOTPEX_H_B,
            riscv_instr::VFDOTPEX_H_R_B: begin
                fpu_op_o = fpnew_pkg::SDOTP;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFDOTPEX_H_R_B}) operand_src_o[2] = RegBRep;
            end
            riscv_instr::VFNDOTPEX_H_B,
            riscv_instr::VFNDOTPEX_H_R_B: begin
                fpu_op_o = fpnew_pkg::SDOTP;
                operand_src_o[0] = RegA;
                operand_src_o[1] = RegB;
                operand_src_o[2] = RegDest;
                op_mode_o      = 1'b1;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFNDOTPEX_H_R_B}) operand_src_o[2] = RegBRep;
            end
            riscv_instr::VFSUMEX_H_B,
            riscv_instr::VFNSUMEX_H_B: begin
                fpu_op_o = fpnew_pkg::EXVSUM;
                operand_src_o[0] = RegA;
                operand_src_o[2] = RegDest;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFNSUMEX_H_B}) op_mode_o = 1'b1;
            end
            // -------------------
            // From float to int
            // -------------------
            // Single Precision Floating-Point
            riscv_instr::FLE_S,
            riscv_instr::FLT_S,
            riscv_instr::FEQ_S: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::FCLASS_S: begin
                fpu_op_o = fpnew_pkg::CLASSIFY;
                operand_src_o[0]   = RegA;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::FCVT_W_S,
            riscv_instr::FCVT_WU_S: begin
                fpu_op_o = fpnew_pkg::F2I;
                operand_src_o[0]   = RegA;
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::FCVT_WU_S}) op_mode_o = 1'b1; // unsigned
            end
            riscv_instr::FMV_X_W: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                fpu_rnd_mode_o   = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                op_mode_o        = 1'b1; // sign-extend result
                operand_src_o[0]   = RegA;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            // Vectorial Single Precision
            riscv_instr::VFEQ_S,
            riscv_instr::VFEQ_R_S: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RDN;
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFEQ_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFNE_S,
            riscv_instr::VFNE_R_S: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RDN;
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                op_mode_o        = 1'b1;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFNE_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFLT_S,
            riscv_instr::VFLT_R_S: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RTZ;
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFLT_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFGE_S,
            riscv_instr::VFGE_R_S: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RTZ;
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                op_mode_o        = 1'b1;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFGE_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFLE_S,
            riscv_instr::VFLE_R_S: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFLE_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFGT_S,
            riscv_instr::VFGT_R_S: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                op_mode_o        = 1'b1;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFGT_R_S}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFCLASS_S: begin
                fpu_op_o = fpnew_pkg::CLASSIFY;
                operand_src_o[0]   = RegA;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP32;
                dst_fmt_o        = fpnew_pkg::FP32;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            // Double Precision Floating-Point
            riscv_instr::FLE_D,
            riscv_instr::FLT_D,
            riscv_instr::FEQ_D: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                src_fmt_o        = fpnew_pkg::FP64;
                dst_fmt_o        = fpnew_pkg::FP64;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::FCLASS_D: begin
                fpu_op_o = fpnew_pkg::CLASSIFY;
                operand_src_o[0]   = RegA;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP64;
                dst_fmt_o        = fpnew_pkg::FP64;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::FCVT_W_D,
            riscv_instr::FCVT_WU_D: begin
                fpu_op_o = fpnew_pkg::F2I;
                operand_src_o[0]   = RegA;
                src_fmt_o        = fpnew_pkg::FP64;
                dst_fmt_o        = fpnew_pkg::FP64;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::FCVT_WU_D}) op_mode_o = 1'b1; // unsigned
            end
            riscv_instr::FMV_X_D: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                fpu_rnd_mode_o   = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o        = fpnew_pkg::FP64;
                dst_fmt_o        = fpnew_pkg::FP64;
                op_mode_o        = 1'b1; // sign-extend result
                operand_src_o[0]   = RegA;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            // [Alternate] Half Precision Floating-Point
            riscv_instr::FLE_H,
            riscv_instr::FLT_H,
            riscv_instr::FEQ_H: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::FCLASS_H: begin
                fpu_op_o = fpnew_pkg::CLASSIFY;
                operand_src_o[0]   = RegA;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::FCVT_W_H,
            riscv_instr::FCVT_WU_H: begin
                fpu_op_o = fpnew_pkg::F2I;
                operand_src_o[0]   = RegA;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::FCVT_WU_H}) op_mode_o = 1'b1; // unsigned
            end
            riscv_instr::FMV_X_H: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                fpu_rnd_mode_o   = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                op_mode_o        = 1'b1; // sign-extend result
                operand_src_o[0]   = RegA;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            // Vectorial [alternate] Half Precision
            riscv_instr::VFEQ_H,
            riscv_instr::VFEQ_R_H: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RDN;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFEQ_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFNE_H,
            riscv_instr::VFNE_R_H: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RDN;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                op_mode_o        = 1'b1;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFNE_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFLT_H,
            riscv_instr::VFLT_R_H: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RTZ;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFLT_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFGE_H,
            riscv_instr::VFGE_R_H: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RTZ;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                op_mode_o        = 1'b1;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFGE_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFLE_H,
            riscv_instr::VFLE_R_H: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFLE_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFGT_H,
            riscv_instr::VFGT_R_H: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                op_mode_o        = 1'b1;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFGT_R_H}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFCLASS_H: begin
                fpu_op_o = fpnew_pkg::CLASSIFY;
                operand_src_o[0]   = RegA;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::VFMV_X_H: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                fpu_rnd_mode_o   = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                op_mode_o        = 1'b1; // sign-extend result
                operand_src_o[0]   = RegA;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::VFCVT_X_H,
            riscv_instr::VFCVT_XU_H: begin
                fpu_op_o = fpnew_pkg::F2I;
                operand_src_o[0]   = RegA;
                src_fmt_o        = fpnew_pkg::FP16;
                dst_fmt_o        = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP16ALT;
                dst_fmt_o      = fpnew_pkg::FP16ALT;
                end
                int_fmt_o        = fpnew_pkg::INT16;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                set_dyn_rm     = 1'b1;
                if (ins_i inside {riscv_instr::VFCVT_XU_H}) op_mode_o = 1'b1; // upper
            end
            // [Alternate] Quarter Precision Floating-Point
            riscv_instr::FLE_B,
            riscv_instr::FLT_B,
            riscv_instr::FEQ_B: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::FCLASS_B: begin
                fpu_op_o = fpnew_pkg::CLASSIFY;
                operand_src_o[0]   = RegA;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::FCVT_W_B,
            riscv_instr::FCVT_WU_B: begin
                fpu_op_o = fpnew_pkg::F2I;
                operand_src_o[0]   = RegA;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::FCVT_WU_B}) op_mode_o = 1'b1; // unsigned
            end
            riscv_instr::FMV_X_B: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                fpu_rnd_mode_o   = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                op_mode_o        = 1'b1; // sign-extend result
                operand_src_o[0]   = RegA;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            // Vectorial Quarter Precision
            riscv_instr::VFEQ_B,
            riscv_instr::VFEQ_R_B: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RDN;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFEQ_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFNE_B,
            riscv_instr::VFNE_R_B: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RDN;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                op_mode_o        = 1'b1;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFNE_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFLT_B,
            riscv_instr::VFLT_R_B: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RTZ;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFLT_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFGE_B,
            riscv_instr::VFGE_R_B: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RTZ;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                op_mode_o        = 1'b1;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFGE_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFLE_B,
            riscv_instr::VFLE_R_B: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFLE_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFGT_B,
            riscv_instr::VFGT_R_B: begin
                fpu_op_o = fpnew_pkg::CMP;
                operand_src_o[0]   = RegA;
                operand_src_o[1]   = RegB;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                op_mode_o        = 1'b1;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                if (ins_i inside {riscv_instr::VFGT_R_B}) operand_src_o[1] = RegBRep;
            end
            riscv_instr::VFCLASS_B: begin
                fpu_op_o = fpnew_pkg::CLASSIFY;
                operand_src_o[0]   = RegA;
                fpu_rnd_mode_o   = fpnew_pkg::RNE;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::VFMV_X_B: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                fpu_rnd_mode_o   = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                op_mode_o        = 1'b1; // sign-extend result
                operand_src_o[0]   = RegA;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
            end
            riscv_instr::VFCVT_X_B,
            riscv_instr::VFCVT_XU_B: begin
                fpu_op_o = fpnew_pkg::F2I;
                operand_src_o[0]   = RegA;
                src_fmt_o        = fpnew_pkg::FP8;
                dst_fmt_o        = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.src == 1'b1) begin
                src_fmt_o      = fpnew_pkg::FP8ALT;
                dst_fmt_o      = fpnew_pkg::FP8ALT;
                end
                int_fmt_o        = fpnew_pkg::INT8;
                vectoral_op_o   = 1'b1;
                rd_is_acc_o = 1'b1;
                rd_is_fpr_o       = 1'b0;
                set_dyn_rm     = 1'b1;
                if (ins_i inside {riscv_instr::VFCVT_XU_B}) op_mode_o = 1'b1; // upper
            end
            // -------------------
            // From int to float
            // -------------------
            // Single Precision Floating-Point
            riscv_instr::FMV_W_X: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = AccBus;
                fpu_rnd_mode_o = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o      = fpnew_pkg::FP32;
                dst_fmt_o      = fpnew_pkg::FP32;
            end
            riscv_instr::FCVT_S_W,
            riscv_instr::FCVT_S_WU: begin
                fpu_op_o = fpnew_pkg::I2F;
                operand_src_o[0] = AccBus;
                dst_fmt_o      = fpnew_pkg::FP32;
                if (ins_i inside {riscv_instr::FCVT_S_WU}) op_mode_o = 1'b1; // unsigned
            end
            // Double Precision Floating-Point
            riscv_instr::FCVT_D_W,
            riscv_instr::FCVT_D_WU: begin
                fpu_op_o = fpnew_pkg::I2F;
                operand_src_o[0] = AccBus;
                src_fmt_o      = fpnew_pkg::FP64;
                dst_fmt_o      = fpnew_pkg::FP64;
                if (ins_i inside {riscv_instr::FCVT_D_WU}) op_mode_o = 1'b1; // unsigned
            end
            // [Alternate] Half Precision Floating-Point
            riscv_instr::FMV_H_X: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = AccBus;
                fpu_rnd_mode_o = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
            end
            riscv_instr::FCVT_H_W,
            riscv_instr::FCVT_H_WU: begin
                fpu_op_o = fpnew_pkg::I2F;
                operand_src_o[0] = AccBus;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                if (ins_i inside {riscv_instr::FCVT_H_WU}) op_mode_o = 1'b1; // unsigned
            end
            // Vectorial Half Precision Floating-Point
            riscv_instr::VFMV_H_X: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = AccBus;
                fpu_rnd_mode_o = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                vectoral_op_o = 1'b1;
            end
            riscv_instr::VFCVT_H_X,
            riscv_instr::VFCVT_H_XU: begin
                fpu_op_o = fpnew_pkg::I2F;
                operand_src_o[0] = AccBus;
                src_fmt_o      = fpnew_pkg::FP16;
                dst_fmt_o      = fpnew_pkg::FP16;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP16ALT;
                dst_fmt_o    = fpnew_pkg::FP16ALT;
                end
                int_fmt_o      = fpnew_pkg::INT16;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCVT_H_XU}) op_mode_o = 1'b1; // upper
            end
            // [Alternate] Quarter Precision Floating-Point
            riscv_instr::FMV_B_X: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = AccBus;
                fpu_rnd_mode_o = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (fpu_fmt_mode_i.dst == 1'b1) begin
                src_fmt_o    = fpnew_pkg::FP8ALT;
                dst_fmt_o    = fpnew_pkg::FP8ALT;
                end
            end
            riscv_instr::FCVT_B_W,
            riscv_instr::FCVT_B_WU: begin
                fpu_op_o = fpnew_pkg::I2F;
                operand_src_o[0] = AccBus;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                if (ins_i inside {riscv_instr::FCVT_B_WU}) op_mode_o = 1'b1; // unsigned
            end
            // Vectorial Quarter Precision Floating-Point
            riscv_instr::VFMV_B_X: begin
                fpu_op_o = fpnew_pkg::SGNJ;
                operand_src_o[0] = AccBus;
                fpu_rnd_mode_o = fpnew_pkg::RUP; // passthrough without checking nan-box
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                vectoral_op_o = 1'b1;
            end
            riscv_instr::VFCVT_B_X,
            riscv_instr::VFCVT_B_XU: begin
                fpu_op_o = fpnew_pkg::I2F;
                operand_src_o[0] = AccBus;
                src_fmt_o      = fpnew_pkg::FP8;
                dst_fmt_o      = fpnew_pkg::FP8;
                int_fmt_o      = fpnew_pkg::INT8;
                vectoral_op_o = 1'b1;
                set_dyn_rm   = 1'b1;
                if (ins_i inside {riscv_instr::VFCVT_B_XU}) op_mode_o = 1'b1; // upper
            end
            // -------------
            // Load / Store
            // -------------
            // Single Precision Floating-Point
            riscv_instr::FLW: begin
                is_load_op_o = 1'b1;
                is_fpu_op_o = 1'b0;
            end
            riscv_instr::FSW: begin
                is_store_op_o = 1'b1;
                operand_src_o[1] = RegB;
                is_fpu_op_o = 1'b0;
                rd_is_fpr_o = 1'b0;
            end
            // Double Precision Floating-Point
            riscv_instr::FLD: begin
                is_load_op_o = 1'b1;
                ls_size_o = DoubleWord;
                is_fpu_op_o = 1'b0;
            end
            riscv_instr::FSD: begin
                is_store_op_o = 1'b1;
                operand_src_o[1] = RegB;
                ls_size_o = DoubleWord;
                is_fpu_op_o = 1'b0;
                rd_is_fpr_o = 1'b0;
            end
            // [Alternate] Half Precision Floating-Point
            riscv_instr::FLH: begin
                is_load_op_o = 1'b1;
                ls_size_o = HalfWord;
                is_fpu_op_o = 1'b0;
            end
            riscv_instr::FSH: begin
                is_store_op_o = 1'b1;
                operand_src_o[1] = RegB;
                ls_size_o = HalfWord;
                is_fpu_op_o = 1'b0;
                rd_is_fpr_o = 1'b0;
            end
            // [Alternate] Quarter Precision Floating-Point
            riscv_instr::FLB: begin
                is_load_op_o = 1'b1;
                ls_size_o = Byte;
                is_fpu_op_o = 1'b0;
            end
            riscv_instr::FSB: begin
                is_store_op_o = 1'b1;
                operand_src_o[1] = RegB;
                ls_size_o = Byte;
                is_fpu_op_o = 1'b0;
                rd_is_fpr_o = 1'b0;
            end
            // -------------
            // CSR Handling
            // -------------
            // Set or clear corresponding CSR
            riscv_instr::CSRRSI: begin
                is_fpu_op_o = 1'b0;
                rd_is_fpr_o = 1'b0;
                is_csr_ins_o = 1'b1;
                // ssr_active_d |= rs1[0];
            end
            riscv_instr::CSRRCI: begin
                is_fpu_op_o = 1'b0;
                rd_is_fpr_o = 1'b0;
                is_csr_ins_o = 1'b1;
                // ssr_active_d &= ~rs1[0];
            end
            default: begin
                is_fpu_op_o = 1'b0;
                error_o = 1'b1;
                rd_is_fpr_o = 1'b0;
            end
        endcase
        // fix round mode for vectors and fp16alt
        if (set_dyn_rm) fpu_rnd_mode_o = fpu_rnd_mode_i;
        // check if src_fmt_o or dst_fmt_o is acutually the alternate version
        // single-format float operations ignore fpu_fmt_mode_i.src
        // reason: for performance reasons when mixing expanding and non-expanding operations
        if (src_fmt_o == fpnew_pkg::FP16 && fpu_fmt_mode_i.src == 1'b1) src_fmt_o = fpnew_pkg::FP16ALT;
        if (dst_fmt_o == fpnew_pkg::FP16 && fpu_fmt_mode_i.dst == 1'b1) dst_fmt_o = fpnew_pkg::FP16ALT;
        if (src_fmt_o == fpnew_pkg::FP8 && fpu_fmt_mode_i.src == 1'b1) src_fmt_o = fpnew_pkg::FP8ALT;
        if (dst_fmt_o == fpnew_pkg::FP8 && fpu_fmt_mode_i.dst == 1'b1) dst_fmt_o = fpnew_pkg::FP8ALT;

        // determine which registers must be loaded
        fpr_srcs_en_o[0] = (op_select[0] == RegA) | (op_select[1] == RegA);
        fpr_srcs_en_o[1] = (op_select[1] == RegB) | (op_select[2] == RegB);
        fpr_srcs_en_o[2] = (op_select[2] == RegC);

        // coallesce duplicate registers
        if (fpr_srcs_en_o[0] & (rs2 == rs1)) begin
            fpr_srcs_en_o[1] = 0;
        end

        if (fpr_srcs_en_o[0] & (rs3 == rs1)) begin
            fpr_srcs_en_o[2] = 0;
        end

        if (fpr_srcs_en_o[1] & (rs3 == rs2)) begin
            fpr_srcs_en_o[2] = 0;
        end
  end
endmodule;