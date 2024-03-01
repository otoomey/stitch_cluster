module stream_merge #(
  /// Number of input streams
  parameter int unsigned N_INP = 32'd0 // Synopsys DC requires a default value for parameters.
) (
  /// Input streams valid handshakes
  input  logic [N_INP-1:0] inp_valid_i,
  /// Input streams ready handshakes
  output logic [N_INP-1:0] inp_ready_o,
  /// Selection mask for the output handshake
  input  logic [N_INP-1:0] sel_i,
  /// Output stream valid handshake
  output logic             oup_valid_o,
  /// Output stream ready handshake
  input  logic             oup_ready_i
);

  // Corner case when `sel_i` is all 0s should not generate valid
  assign oup_valid_o = &(inp_valid_i | ~sel_i) && |sel_i;
  for (genvar i = 0; i < N_INP; i++) begin : gen_inp_ready
    assign inp_ready_o[i] = oup_valid_o & oup_ready_i & sel_i[i];
  end

endmodule