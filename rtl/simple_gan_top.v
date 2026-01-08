// ============================================================================
// Simple-GAN Top Module (Q1.15 Fixed-Point)
// ============================================================================
// Purpose: Top-level module integrating Generator and Discriminator
// Control: Simple FSM (IDLE -> COMPUTE_GEN -> COMPUTE_DISC -> DONE)
//
// This is a NON-shared hardware implementation:
//   - Generator and Discriminator run sequentially
//   - Each network is fully combinational
//   - FSM sequences: Gen (1 cycle) -> Disc (1 cycle)
//   - Total latency: 2 clock cycles
//
// For shared hardware alternative, see comments at end of file
// ============================================================================

`timescale 1ns/1ps

module simple_gan_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,           // Start computation
    
    // Generator inputs
    input  wire signed [15:0] noise_0,
    input  wire signed [15:0] noise_1,
    
    // Generator outputs (9D image)
    output reg  signed [15:0] gen_image_0,
    output reg  signed [15:0] gen_image_1,
    output reg  signed [15:0] gen_image_2,
    output reg  signed [15:0] gen_image_3,
    output reg  signed [15:0] gen_image_4,
    output reg  signed [15:0] gen_image_5,
    output reg  signed [15:0] gen_image_6,
    output reg  signed [15:0] gen_image_7,
    output reg  signed [15:0] gen_image_8,
    
    // Discriminator output
    output reg  signed [15:0] disc_prob,
    
    // Control outputs
    output reg         done,
    output reg         gen_valid,       // Generator output valid
    output reg         disc_valid       // Discriminator output valid
);

    // ========================================================================
    // FSM States
    // ========================================================================
    localparam IDLE         = 3'b000;
    localparam COMPUTE_GEN  = 3'b001;
    localparam COMPUTE_DISC = 3'b010;
    localparam DONE_STATE   = 3'b011;
    
    reg [2:0] state, next_state;

    // ========================================================================
    // Generator Instantiation
    // ========================================================================
    wire signed [15:0] gen_out_0, gen_out_1, gen_out_2;
    wire signed [15:0] gen_out_3, gen_out_4, gen_out_5;
    wire signed [15:0] gen_out_6, gen_out_7, gen_out_8;
    
    generator_q15 u_generator (
        .noise_0  (noise_0),
        .noise_1  (noise_1),
        .image_0  (gen_out_0),
        .image_1  (gen_out_1),
        .image_2  (gen_out_2),
        .image_3  (gen_out_3),
        .image_4  (gen_out_4),
        .image_5  (gen_out_5),
        .image_6  (gen_out_6),
        .image_7  (gen_out_7),
        .image_8  (gen_out_8)
    );

    // ========================================================================
    // Discriminator Instantiation
    // ========================================================================
    wire signed [15:0] disc_out;
    
    discriminator_q15 u_discriminator (
        .image_0  (gen_image_0),
        .image_1  (gen_image_1),
        .image_2  (gen_image_2),
        .image_3  (gen_image_3),
        .image_4  (gen_image_4),
        .image_5  (gen_image_5),
        .image_6  (gen_image_6),
        .image_7  (gen_image_7),
        .image_8  (gen_image_8),
        .prob     (disc_out)
    );

    // ========================================================================
    // FSM: State Register
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // ========================================================================
    // FSM: Next State Logic
    // ========================================================================
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start)
                    next_state = COMPUTE_GEN;
            end
            
            COMPUTE_GEN: begin
                next_state = COMPUTE_DISC;
            end
            
            COMPUTE_DISC: begin
                next_state = DONE_STATE;
            end
            
            DONE_STATE: begin
                if (!start)
                    next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end

    // ========================================================================
    // FSM: Output Logic and Datapath Control
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gen_image_0 <= 16'h0;
            gen_image_1 <= 16'h0;
            gen_image_2 <= 16'h0;
            gen_image_3 <= 16'h0;
            gen_image_4 <= 16'h0;
            gen_image_5 <= 16'h0;
            gen_image_6 <= 16'h0;
            gen_image_7 <= 16'h0;
            gen_image_8 <= 16'h0;
            disc_prob   <= 16'h0;
            done        <= 1'b0;
            gen_valid   <= 1'b0;
            disc_valid  <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done        <= 1'b0;
                    gen_valid   <= 1'b0;
                    disc_valid  <= 1'b0;
                end
                
                COMPUTE_GEN: begin
                    // Capture generator output
                    gen_image_0 <= gen_out_0;
                    gen_image_1 <= gen_out_1;
                    gen_image_2 <= gen_out_2;
                    gen_image_3 <= gen_out_3;
                    gen_image_4 <= gen_out_4;
                    gen_image_5 <= gen_out_5;
                    gen_image_6 <= gen_out_6;
                    gen_image_7 <= gen_out_7;
                    gen_image_8 <= gen_out_8;
                    gen_valid   <= 1'b1;
                end
                
                COMPUTE_DISC: begin
                    // Capture discriminator output
                    disc_prob   <= disc_out;
                    disc_valid  <= 1'b1;
                end
                
                DONE_STATE: begin
                    done <= 1'b1;
                end
            endcase
        end
    end

endmodule

// ============================================================================
// SHARED HARDWARE ALTERNATIVE (NOT IMPLEMENTED)
// ============================================================================
// For this tiny network (63 total weights), shared hardware adds complexity
// without significant benefit. However, if needed, here's the approach:
//
// 1. Create a generic fc_neuron_q15.v module:
//    - Parameters: INPUT_SIZE, enable signal
//    - Shared MAC loop or parallel multipliers
//    - Activation select (tanh/sigmoid)
//
// 2. Weight memory controller:
//    - Single weight RAM with address decoder
//    - Mux between Wg2, Wg3, Wd2, Wd3 based on FSM state
//
// 3. Extended FSM:
//    IDLE -> LOAD_WG2 -> COMPUTE_G2 -> LOAD_WG3 -> COMPUTE_G3 ->
//    LOAD_WD2 -> COMPUTE_D2 -> LOAD_WD3 -> COMPUTE_D3 -> DONE
//
// 4. Trade-offs:
//    + Area: ~30% reduction (estimated)
//    - Latency: 8-10 cycles vs 2 cycles
//    - Complexity: 3x more control logic
//    - Power: Similar (more cycles vs less combinational)
//
// Recommendation: For LSI Contest with <100 weights, non-shared is better.
// ============================================================================
