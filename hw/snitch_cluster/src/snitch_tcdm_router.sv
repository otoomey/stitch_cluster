// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Florian Zaruba <zarubaf@iis.ee.ethz.ch>
// Author: Wolfgang Roenninger <wroennin@ethz.ch>

`include "mem_interface/typedef.svh"

/// Lightweight wrapper for a fixed response latency router, i.e.,
/// something that can be used to router memories.
module snitch_tcdm_router #(
  parameter int unsigned AddrWidth             = 32,
  /// Number of inputs into the router (`> 0`).
  parameter int unsigned NumInp                = 32'd5,
  /// Number of outputs from the router (`> 0`).
  parameter int unsigned NumOut                = 32'd10,
  /// Radix of the individual switch points of the network.
  /// Currently supported are `32'd2` and `32'd4`.
  parameter int unsigned Radix                 = 32'd2,
  /// Payload type of the data request ports.
  parameter type         tcdm_req_t            = logic,
  /// Payload type of the data response ports.
  parameter type         tcdm_rsp_t            = logic,
  parameter int unsigned MemCoallWidth         = 0,
  /// Data size of the router. Only the data portion counts. The offsets
  /// into the address are derived from this.
  parameter int unsigned DataWidth             = 32,
  /// Additional user payload to route.
  parameter type         user_t                = logic
) (
  /// Clock, positive edge triggered.
  input  logic                             clk_i,
  /// Reset, active low.
  input  logic                             rst_ni,
  /// Input request port.
  input  tcdm_req_t           [NumInp-1:0] mst_req_i,
  /// Input response port.
  output tcdm_rsp_t           [NumInp-1:0] mst_rsp_o,
  /// Output request port.
  output tcdm_req_t           [NumOut-1:0] agnt_req_o,
  /// Output response port.
  input  tcdm_rsp_t           [NumOut-1:0] agnt_rsp_i
);

  localparam int unsigned ByteOffset = $clog2(DataWidth/8);

  // Width of the bank select signal.
  localparam int unsigned SelWidth = cf_math_pkg::idx_width(NumOut);
  typedef logic [SelWidth-1:0] select_t;
  select_t [NumInp-1:0] bank_select;

  typedef struct packed {
    // Which bank was selected.
    select_t bank_select;
  } rsp_t;

  // Generate the `bank_select` signal based on the address.
  // This generates a bank interleaved addressing scheme, where consecutive
  // addresses are routed to individual banks.
  for (genvar i = 0; i < NumInp; i++) begin : gen_bank_select
    assign bank_select[i] = mst_req_i[i].q.addr[ByteOffset+MemCoallWidth+:SelWidth];
  end

  tcdm_req_t [NumInp-1:0] payload_req_i;
  tcdm_req_t [NumInp-1:0] payload_req_o;

  logic [NumInp-1:0] mst_req_q_valid_flat, mst_rsp_q_ready_flat;
  logic [NumOut-1:0] agnt_req_q_valid_flat, agnt_req_q_ready_flat;

  // The usual struct packing unpacking.
  for (genvar i = 0; i < NumInp; i++) begin : gen_flat_inp
    assign mst_req_q_valid_flat[i] = mst_req_i[i].q_valid;
    assign mst_rsp_o[i].q_ready = mst_rsp_q_ready_flat[i];
    assign payload_req_i[i].q = '{
      addr: {
        mst_req_i[i].q.addr[AddrWidth-1:ByteOffset+MemCoallWidth+SelWidth], // 9
        mst_req_i[i].q.addr[ByteOffset+MemCoallWidth-1:0] // 6
      },
      write: mst_req_i[i].q.write,
      amo: mst_req_i[i].q.amo,
      data: mst_req_i[i].q.data,
      strb: mst_req_i[i].q.strb,
      user: mst_req_i[i].q.user
    };
  end

  for (genvar i = 0; i < NumOut; i++) begin : gen_flat_oup
    assign agnt_req_o[i].q_valid = agnt_req_q_valid_flat[i];
    assign agnt_req_o[i].q = payload_req_o[i].q;
    assign agnt_req_q_ready_flat[i] = agnt_rsp_i[i].q_ready;
  end

  // ------------
  // Request Side
  // ------------
  // We need to arbitrate the requests coming from the input side and resolve
  // potential bank conflicts. Therefore a full arbitration tree is needed.
  stream_xbar #(
    .NumInp      ( NumInp     ),
    .NumOut      ( NumOut     ),
    .payload_t   ( tcdm_req_t ),
    .OutSpillReg ( 1'b0       ),
    .ExtPrio     ( 1'b0       ),
    .AxiVldRdy   ( 1'b1       ),
    .LockIn      ( 1'b1       )
  ) i_stream_xbar (
    .clk_i,
    .rst_ni,
    .flush_i ( 1'b0 ),
    .rr_i    ( '0 ),
    .data_i  ( payload_req_i ),
    .sel_i   ( bank_select ),
    .valid_i ( mst_req_q_valid_flat ),
    .ready_o ( mst_rsp_q_ready_flat ),
    .data_o  ( payload_req_o        ),
    .idx_o   ( ),
    .valid_o ( agnt_req_q_valid_flat ),
    .ready_i ( agnt_req_q_ready_flat )
  );
 
  // -------------
  // Response Side
  // -------------
  logic [NumInp-1:0] rsp_valid;
  for (genvar i = 0; i < NumInp; i++) begin : gen_rsp_mux
    rsp_t out_rsp_mux, in_rsp_mux;
    assign in_rsp_mux = '{
      bank_select: bank_select[i]
    };
    shift_reg_gated #(
      .dtype ( rsp_t ),
      .Depth ( 1 )
    ) i_shift_reg (
      .clk_i,
      .rst_ni,
      .valid_i ( mst_req_i[i].q_valid & mst_rsp_o[i].q_ready ),
      .data_i  ( in_rsp_mux ),
      .valid_o ( rsp_valid[i] ),
      .data_o  ( out_rsp_mux )
    );
    assign mst_rsp_o[i].p.data = agnt_rsp_i[out_rsp_mux.bank_select].p.data;
    assign mst_rsp_o[i].p_valid = rsp_valid[i];
  end

  // initial begin
  //   $display("addr_width: %d, mem_coall: %d, ByteOffset: %d, SelWidth: %d", AddrWidth, MemCoallWidth, ByteOffset, SelWidth);
  //   $display("mem_high_width: %d", $bits(mst_req_i[0].q.addr[AddrWidth-1:ByteOffset+MemCoallWidth+SelWidth]));
  //   $display("mem_low_width: %d", $bits(mst_req_i[0].q.addr[ByteOffset+MemCoallWidth-1:0]));
  //   $display("mst addr width: %d", $bits(mst_req_i[0].q.addr));
  //   $display("payload addr width: %d", $bits(payload_req_i[0].q.addr));
  // end
  
  // always @(posedge clk_i) begin
  //   // if a transactions is happening
  //   if (|mst_req_q_valid_flat & |mst_rsp_q_ready_flat) begin
  //     $write("router: ");
  //   end
  //   foreach (bank_select[i]) begin
  //     if (mst_req_q_valid_flat[i] && mst_rsp_q_ready_flat[i]) begin
  //       $write("[%2d:%x]->[%2d:%x] ", i, mst_req_i[i].q.addr, bank_select[i], payload_req_i[i].q.addr);
  //     end
  //   end
  //   if (|mst_req_q_valid_flat & |mst_rsp_q_ready_flat) begin
  //     $write("\n");
  //   end
  //   if (|rsp_valid) begin
  //     $display("- router: response: %b", rsp_valid);
  //   end
  // end
endmodule
