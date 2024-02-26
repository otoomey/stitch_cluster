// modified fifo_v3; resets to some defined input
module snitch_sb_ipool #(
    parameter int unsigned Depth        = 8,    // depth can be arbitrary from 0 to 2**32
    parameter [Depth-1:0] ResetState [Depth-1:0] = '0,
    // DO NOT OVERWRITE THIS PARAMETER
    parameter type dtype                = logic [$clog2(Depth)-1:0],
    parameter int unsigned AddrDepth   = (Depth > 1) ? $clog2(Depth) : 1
)(
    input  logic  clk_i,                // Clock
    input  logic  rst_ni,               // Asynchronous reset active low
    // status flags
    output logic  full_o,           // queue is full
    output logic  empty_o,          // queue is empty
    output logic  [AddrDepth-1:0] usage_o,  // fill pointer
    // as long as the queue is not full we can push new data
    input  dtype  data_i,           // data to push into the queue
    input  logic  push_i,           // data is valid and can be pushed to the queue
    // as long as the queue is not empty we can pop new elements
    output dtype  data_o,           // output data
    input  logic  pop_i             // pop head from queue
);
    // local parameter
    // FIFO depth - handle the case of pass-through, synthesizer will do constant propagation
    localparam int unsigned FifoDepth = (Depth > 0) ? Depth : 1;
    // clock gating control
    logic gate_clock;
    // pointer to the read and write section of the queue
    logic [AddrDepth - 1:0] read_pointer_n, read_pointer_q, write_pointer_n, write_pointer_q;
    // keep a counter to keep track of the current queue status
    // this integer will be truncated by the synthesis tool
    logic [AddrDepth:0] status_cnt_n, status_cnt_q;
    // actual memory
    dtype [FifoDepth - 1:0] mem_n, mem_q;

    assign usage_o = status_cnt_q[AddrDepth-1:0];

    if (Depth == 0) begin : gen_pass_through
        assign empty_o     = ~push_i;
        assign full_o      = ~pop_i;
    end else begin : gen_fifo
        assign full_o       = (status_cnt_q == FifoDepth[AddrDepth:0]);
        assign empty_o      = (status_cnt_q == 0);
    end
    // status flags

    // read and write queue logic
    always_comb begin : read_write_comb
        // default assignment
        read_pointer_n  = read_pointer_q;
        write_pointer_n = write_pointer_q;
        status_cnt_n    = status_cnt_q;
        data_o          = (Depth == 0) ? data_i : mem_q[read_pointer_q];
        mem_n           = mem_q;
        gate_clock      = 1'b1;

        // push a new element to the queue
        if (push_i && ~full_o) begin
            // push the data onto the queue
            mem_n[write_pointer_q] = data_i;
            // un-gate the clock, we want to write something
            gate_clock = 1'b0;
            // increment the write counter
            // this is dead code when Depth is a power of two
            if (write_pointer_q == FifoDepth[AddrDepth-1:0] - 1)
                write_pointer_n = '0;
            else
                write_pointer_n = write_pointer_q + 1;
            // increment the overall counter
            status_cnt_n    = status_cnt_q + 1;
        end

        if (pop_i && ~empty_o) begin
            // read from the queue is a default assignment
            // but increment the read pointer...
            // this is dead code when Depth is a power of two
            if (read_pointer_n == FifoDepth[AddrDepth-1:0] - 1)
                read_pointer_n = '0;
            else
                read_pointer_n = read_pointer_q + 1;
            // ... and decrement the overall count
            status_cnt_n   = status_cnt_q - 1;
        end

        // keep the count pointer stable if we push and pop at the same time
        if (push_i && pop_i &&  ~full_o && ~empty_o)
            status_cnt_n   = status_cnt_q;
    end

    // sequential process
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if(~rst_ni) begin
            read_pointer_q  <= '0;
            write_pointer_q <= '0;
            status_cnt_q    <= FifoDepth[AddrDepth:0];
        end else begin
            read_pointer_q  <= read_pointer_n;
            write_pointer_q <= write_pointer_n;
            status_cnt_q    <= status_cnt_n;
        end
    end

    for (genvar i = 0; i < Depth; i++) begin
        always_ff @(posedge clk_i or negedge rst_ni) begin
            if(~rst_ni) begin
                mem_q[i] <= i;
            end else if (!gate_clock) begin
                mem_q[i] <= mem_n[i];
            end
        end
    end

    // needed for simulation
    for (genvar i = 0; i < Depth; i++) begin
        initial begin
            mem_q[i] = i;
            read_pointer_q = '0;
            write_pointer_q = '0;
            status_cnt_q    = FifoDepth[AddrDepth:0];
            // $display("mem_q[%d]=%d", i, mem_q[i]);
        end
    end
endmodule