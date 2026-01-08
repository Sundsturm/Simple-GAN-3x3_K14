// ============================================================================
// Discriminator Network (Q1.15 Fixed-Point)
// ============================================================================
// Purpose: Simple-GAN Discriminator for real/fake classification
// Input:   image[8:0] - 9D input vector (3x3 image, Q1.15)
// Output:  prob - probability of real image (Q1.15, range [0,1])
//
// Architecture:
//   Layer 2: FC(9->3) + tanh
//   Layer 3: FC(3->1) + sigmoid
//
// Weights: Wd2[3x9], bd2[3], Wd3[1x3], bd3[1]
// Implementation: Pure combinational (1 cycle latency)
// ============================================================================

`timescale 1ns/1ps

module discriminator_q15 (
    input  wire signed [15:0] image_0,      // input pixel [0]
    input  wire signed [15:0] image_1,      // input pixel [1]
    input  wire signed [15:0] image_2,      // input pixel [2]
    input  wire signed [15:0] image_3,      // input pixel [3]
    input  wire signed [15:0] image_4,      // input pixel [4]
    input  wire signed [15:0] image_5,      // input pixel [5]
    input  wire signed [15:0] image_6,      // input pixel [6]
    input  wire signed [15:0] image_7,      // input pixel [7]
    input  wire signed [15:0] image_8,      // input pixel [8]
    output wire signed [15:0] prob          // real/fake probability
);

    // ========================================================================
    // Weight and Bias Memories
    // ========================================================================
    // Wd2: 3x9 = 27 weights, bd2: 3 biases
    // Wd3: 1x3 = 3 weights, bd3: 1 bias
    
    reg signed [15:0] Wd2 [0:26];  // Layer 2 weights
    reg signed [15:0] bd2 [0:2];   // Layer 2 biases
    reg signed [15:0] Wd3 [0:2];   // Layer 3 weights
    reg signed [15:0] bd3 [0:0];   // Layer 3 bias
    
    initial begin
        $readmemh("parameters/Wd2.txt", Wd2);
        $readmemh("parameters/bd2.txt", bd2);
        $readmemh("parameters/Wd3.txt", Wd3);
        $readmemh("parameters/bd3.txt", bd3);
    end

    // ========================================================================
    // Layer 2: FC(9->3) + tanh
    // ========================================================================
    // Hidden layer: h[i] = tanh(Wd2[i,:] * image + bd2[i])
    // Wd2 storage: row-major [row0_col0, row0_col1, ..., row0_col8, row1_col0, ...]
    
    wire signed [31:0] z2_0_p0, z2_0_p1, z2_0_p2, z2_0_p3, z2_0_p4, z2_0_p5, z2_0_p6, z2_0_p7, z2_0_p8;
    wire signed [31:0] z2_1_p0, z2_1_p1, z2_1_p2, z2_1_p3, z2_1_p4, z2_1_p5, z2_1_p6, z2_1_p7, z2_1_p8;
    wire signed [31:0] z2_2_p0, z2_2_p1, z2_2_p2, z2_2_p3, z2_2_p4, z2_2_p5, z2_2_p6, z2_2_p7, z2_2_p8;
    
    wire signed [15:0] z2_0_sum, z2_1_sum, z2_2_sum;
    wire signed [15:0] h0, h1, h2;
    
    // Neuron 0: z2[0] = Wd2[0,:] * image + bd2[0]
    assign z2_0_p0 = Wd2[0] * image_0;
    assign z2_0_p1 = Wd2[1] * image_1;
    assign z2_0_p2 = Wd2[2] * image_2;
    assign z2_0_p3 = Wd2[3] * image_3;
    assign z2_0_p4 = Wd2[4] * image_4;
    assign z2_0_p5 = Wd2[5] * image_5;
    assign z2_0_p6 = Wd2[6] * image_6;
    assign z2_0_p7 = Wd2[7] * image_7;
    assign z2_0_p8 = Wd2[8] * image_8;
    assign z2_0_sum = z2_0_p0[30:15] + z2_0_p1[30:15] + z2_0_p2[30:15] + 
                      z2_0_p3[30:15] + z2_0_p4[30:15] + z2_0_p5[30:15] +
                      z2_0_p6[30:15] + z2_0_p7[30:15] + z2_0_p8[30:15] + bd2[0];
    
    // Neuron 1: z2[1] = Wd2[1,:] * image + bd2[1]
    assign z2_1_p0 = Wd2[9] * image_0;
    assign z2_1_p1 = Wd2[10] * image_1;
    assign z2_1_p2 = Wd2[11] * image_2;
    assign z2_1_p3 = Wd2[12] * image_3;
    assign z2_1_p4 = Wd2[13] * image_4;
    assign z2_1_p5 = Wd2[14] * image_5;
    assign z2_1_p6 = Wd2[15] * image_6;
    assign z2_1_p7 = Wd2[16] * image_7;
    assign z2_1_p8 = Wd2[17] * image_8;
    assign z2_1_sum = z2_1_p0[30:15] + z2_1_p1[30:15] + z2_1_p2[30:15] + 
                      z2_1_p3[30:15] + z2_1_p4[30:15] + z2_1_p5[30:15] +
                      z2_1_p6[30:15] + z2_1_p7[30:15] + z2_1_p8[30:15] + bd2[1];
    
    // Neuron 2: z2[2] = Wd2[2,:] * image + bd2[2]
    assign z2_2_p0 = Wd2[18] * image_0;
    assign z2_2_p1 = Wd2[19] * image_1;
    assign z2_2_p2 = Wd2[20] * image_2;
    assign z2_2_p3 = Wd2[21] * image_3;
    assign z2_2_p4 = Wd2[22] * image_4;
    assign z2_2_p5 = Wd2[23] * image_5;
    assign z2_2_p6 = Wd2[24] * image_6;
    assign z2_2_p7 = Wd2[25] * image_7;
    assign z2_2_p8 = Wd2[26] * image_8;
    assign z2_2_sum = z2_2_p0[30:15] + z2_2_p1[30:15] + z2_2_p2[30:15] + 
                      z2_2_p3[30:15] + z2_2_p4[30:15] + z2_2_p5[30:15] +
                      z2_2_p6[30:15] + z2_2_p7[30:15] + z2_2_p8[30:15] + bd2[2];
    
    // Activation: tanh
    tanh_approx_q15 tanh_h0 (.x(z2_0_sum), .y(h0));
    tanh_approx_q15 tanh_h1 (.x(z2_1_sum), .y(h1));
    tanh_approx_q15 tanh_h2 (.x(z2_2_sum), .y(h2));

    // ========================================================================
    // Layer 3: FC(3->1) + sigmoid
    // ========================================================================
    // Output: prob = sigmoid(Wd3 * h + bd3)
    
    wire signed [31:0] z3_p0, z3_p1, z3_p2;
    wire signed [15:0] z3_sum;
    
    assign z3_p0 = Wd3[0] * h0;
    assign z3_p1 = Wd3[1] * h1;
    assign z3_p2 = Wd3[2] * h2;
    assign z3_sum = z3_p0[30:15] + z3_p1[30:15] + z3_p2[30:15] + bd3[0];
    
    // Activation: sigmoid
    sigmoid_approx_q15 sigmoid_out (.x(z3_sum), .y(prob));

endmodule
