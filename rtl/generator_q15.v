// ============================================================================
// Generator Network (Q1.15 Fixed-Point)
// ============================================================================
// Purpose: Simple-GAN Generator for 3x3 image generation
// Input:   noise[1:0] - 2D latent vector (Q1.15)
// Output:  image[8:0] - 9D output vector (3x3 image, Q1.15)
//
// Architecture:
//   Layer 2: FC(2->3) + tanh
//   Layer 3: FC(3->9) + tanh
//
// Weights: Wg2[3x2], bg2[3], Wg3[9x3], bg3[9]
// Implementation: Pure combinational (1 cycle latency)
// ============================================================================

`timescale 1ns/1ps

module generator_q15 (
    input  wire signed [15:0] noise_0,      // noise[0]
    input  wire signed [15:0] noise_1,      // noise[1]
    output wire signed [15:0] image_0,      // output pixel [0]
    output wire signed [15:0] image_1,      // output pixel [1]
    output wire signed [15:0] image_2,      // output pixel [2]
    output wire signed [15:0] image_3,      // output pixel [3]
    output wire signed [15:0] image_4,      // output pixel [4]
    output wire signed [15:0] image_5,      // output pixel [5]
    output wire signed [15:0] image_6,      // output pixel [6]
    output wire signed [15:0] image_7,      // output pixel [7]
    output wire signed [15:0] image_8       // output pixel [8]
);

    // ========================================================================
    // Weight and Bias Memories
    // ========================================================================
    // Wg2: 3x2 = 6 weights, bg2: 3 biases
    // Wg3: 9x3 = 27 weights, bg3: 9 biases
    
    reg signed [15:0] Wg2 [0:5];   // Layer 2 weights
    reg signed [15:0] bg2 [0:2];   // Layer 2 biases
    reg signed [15:0] Wg3 [0:26];  // Layer 3 weights
    reg signed [15:0] bg3 [0:8];   // Layer 3 biases
    
    initial begin
        $readmemh("parameters/Wg2_q15.hex", Wg2);
        $readmemh("parameters/bg2_q15.hex", bg2);
        $readmemh("parameters/Wg3_q15.hex", Wg3);
        $readmemh("parameters/bg3_q15.hex", bg3);
    end

    // ========================================================================
    // Layer 2: FC(2->3) + tanh
    // ========================================================================
    // Hidden layer computation: h[i] = tanh(Wg2[i,:] * noise + bg2[i])
    // Wg2 storage: row-major [row0_col0, row0_col1, row1_col0, row1_col1, ...]
    
    wire signed [31:0] z2_0_prod0, z2_0_prod1;
    wire signed [31:0] z2_1_prod0, z2_1_prod1;
    wire signed [31:0] z2_2_prod0, z2_2_prod1;
    
    wire signed [15:0] z2_0_sum, z2_1_sum, z2_2_sum;
    wire signed [15:0] h0, h1, h2;
    
    // Neuron 0: z2[0] = Wg2[0,0]*noise[0] + Wg2[0,1]*noise[1] + bg2[0]
    assign z2_0_prod0 = Wg2[0] * noise_0;
    assign z2_0_prod1 = Wg2[1] * noise_1;
    assign z2_0_sum = z2_0_prod0[30:15] + z2_0_prod1[30:15] + bg2[0];
    
    // Neuron 1: z2[1] = Wg2[1,0]*noise[0] + Wg2[1,1]*noise[1] + bg2[1]
    assign z2_1_prod0 = Wg2[2] * noise_0;
    assign z2_1_prod1 = Wg2[3] * noise_1;
    assign z2_1_sum = z2_1_prod0[30:15] + z2_1_prod1[30:15] + bg2[1];
    
    // Neuron 2: z2[2] = Wg2[2,0]*noise[0] + Wg2[2,1]*noise[1] + bg2[2]
    assign z2_2_prod0 = Wg2[4] * noise_0;
    assign z2_2_prod1 = Wg2[5] * noise_1;
    assign z2_2_sum = z2_2_prod0[30:15] + z2_2_prod1[30:15] + bg2[2];
    
    // Activation: tanh
    tanh_approx_q15 tanh_h0 (.x(z2_0_sum), .y(h0));
    tanh_approx_q15 tanh_h1 (.x(z2_1_sum), .y(h1));
    tanh_approx_q15 tanh_h2 (.x(z2_2_sum), .y(h2));

    // ========================================================================
    // Layer 3: FC(3->9) + tanh
    // ========================================================================
    // Output layer: out[i] = tanh(Wg3[i,:] * h + bg3[i])
    // Wg3 storage: row-major [row0_col0, row0_col1, row0_col2, row1_col0, ...]
    
    wire signed [31:0] z3_0_p0, z3_0_p1, z3_0_p2;
    wire signed [31:0] z3_1_p0, z3_1_p1, z3_1_p2;
    wire signed [31:0] z3_2_p0, z3_2_p1, z3_2_p2;
    wire signed [31:0] z3_3_p0, z3_3_p1, z3_3_p2;
    wire signed [31:0] z3_4_p0, z3_4_p1, z3_4_p2;
    wire signed [31:0] z3_5_p0, z3_5_p1, z3_5_p2;
    wire signed [31:0] z3_6_p0, z3_6_p1, z3_6_p2;
    wire signed [31:0] z3_7_p0, z3_7_p1, z3_7_p2;
    wire signed [31:0] z3_8_p0, z3_8_p1, z3_8_p2;
    
    wire signed [15:0] z3_0, z3_1, z3_2, z3_3, z3_4, z3_5, z3_6, z3_7, z3_8;
    
    // Neuron 0
    assign z3_0_p0 = Wg3[0] * h0;
    assign z3_0_p1 = Wg3[1] * h1;
    assign z3_0_p2 = Wg3[2] * h2;
    assign z3_0 = z3_0_p0[30:15] + z3_0_p1[30:15] + z3_0_p2[30:15] + bg3[0];
    
    // Neuron 1
    assign z3_1_p0 = Wg3[3] * h0;
    assign z3_1_p1 = Wg3[4] * h1;
    assign z3_1_p2 = Wg3[5] * h2;
    assign z3_1 = z3_1_p0[30:15] + z3_1_p1[30:15] + z3_1_p2[30:15] + bg3[1];
    
    // Neuron 2
    assign z3_2_p0 = Wg3[6] * h0;
    assign z3_2_p1 = Wg3[7] * h1;
    assign z3_2_p2 = Wg3[8] * h2;
    assign z3_2 = z3_2_p0[30:15] + z3_2_p1[30:15] + z3_2_p2[30:15] + bg3[2];
    
    // Neuron 3
    assign z3_3_p0 = Wg3[9] * h0;
    assign z3_3_p1 = Wg3[10] * h1;
    assign z3_3_p2 = Wg3[11] * h2;
    assign z3_3 = z3_3_p0[30:15] + z3_3_p1[30:15] + z3_3_p2[30:15] + bg3[3];
    
    // Neuron 4
    assign z3_4_p0 = Wg3[12] * h0;
    assign z3_4_p1 = Wg3[13] * h1;
    assign z3_4_p2 = Wg3[14] * h2;
    assign z3_4 = z3_4_p0[30:15] + z3_4_p1[30:15] + z3_4_p2[30:15] + bg3[4];
    
    // Neuron 5
    assign z3_5_p0 = Wg3[15] * h0;
    assign z3_5_p1 = Wg3[16] * h1;
    assign z3_5_p2 = Wg3[17] * h2;
    assign z3_5 = z3_5_p0[30:15] + z3_5_p1[30:15] + z3_5_p2[30:15] + bg3[5];
    
    // Neuron 6
    assign z3_6_p0 = Wg3[18] * h0;
    assign z3_6_p1 = Wg3[19] * h1;
    assign z3_6_p2 = Wg3[20] * h2;
    assign z3_6 = z3_6_p0[30:15] + z3_6_p1[30:15] + z3_6_p2[30:15] + bg3[6];
    
    // Neuron 7
    assign z3_7_p0 = Wg3[21] * h0;
    assign z3_7_p1 = Wg3[22] * h1;
    assign z3_7_p2 = Wg3[23] * h2;
    assign z3_7 = z3_7_p0[30:15] + z3_7_p1[30:15] + z3_7_p2[30:15] + bg3[7];
    
    // Neuron 8
    assign z3_8_p0 = Wg3[24] * h0;
    assign z3_8_p1 = Wg3[25] * h1;
    assign z3_8_p2 = Wg3[26] * h2;
    assign z3_8 = z3_8_p0[30:15] + z3_8_p1[30:15] + z3_8_p2[30:15] + bg3[8];
    
    // Activation: tanh for all outputs
    tanh_approx_q15 tanh_out0 (.x(z3_0), .y(image_0));
    tanh_approx_q15 tanh_out1 (.x(z3_1), .y(image_1));
    tanh_approx_q15 tanh_out2 (.x(z3_2), .y(image_2));
    tanh_approx_q15 tanh_out3 (.x(z3_3), .y(image_3));
    tanh_approx_q15 tanh_out4 (.x(z3_4), .y(image_4));
    tanh_approx_q15 tanh_out5 (.x(z3_5), .y(image_5));
    tanh_approx_q15 tanh_out6 (.x(z3_6), .y(image_6));
    tanh_approx_q15 tanh_out7 (.x(z3_7), .y(image_7));
    tanh_approx_q15 tanh_out8 (.x(z3_8), .y(image_8));

endmodule
