module stream_reduce #(
  parameter type data_t = logic  // Vivado requires a default value for type parameters.
) (
  input  logic  [1:0]     valid_i,
  output logic  [1:0]     ready_o,

  output logic            valid_o,
  input  logic            ready_i
);

  assign ready_o[0] = ready_i;
  assign ready_o[1] = ready_i;

  assign valid_o  = valid_i[0] | valid_i[1];

endmodule