// ============================================================================
// Simple-GAN Testbench
// ============================================================================
// Purpose: Testbench for Simple-GAN inference
// Tests:
//   1. Generator output for various noise inputs
//   2. Discriminator classification of generated images
//   3. Loading sample inputs from parameter files
//
// Usage:
//   iverilog -o simple_gan_tb.vvp simple_gan_tb.v simple_gan_top.v \
//            generator_q15.v discriminator_q15.v \
//            activation_unit/tanh_approx_q15.v \
//            activation_unit/sigmoid_approx_q15.v
//   vvp simple_gan_tb.vvp
//   gtkwave simple_gan_tb.vcd
// ============================================================================

`timescale 1ns/1ps

module simple_gan_tb;

    // ========================================================================
    // Clock and Reset
    // ========================================================================
    reg clk;
    reg rst_n;
    
    parameter CLK_PERIOD = 10; // 10ns = 100MHz
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // ========================================================================
    // DUT Signals
    // ========================================================================
    reg         start;
    reg  signed [15:0] noise_0, noise_1;
    wire signed [15:0] gen_image_0, gen_image_1, gen_image_2;
    wire signed [15:0] gen_image_3, gen_image_4, gen_image_5;
    wire signed [15:0] gen_image_6, gen_image_7, gen_image_8;
    wire signed [15:0] disc_prob;
    wire        done, gen_valid, disc_valid;

    // ========================================================================
    // DUT Instantiation
    // ========================================================================
    simple_gan_top u_dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .noise_0     (noise_0),
        .noise_1     (noise_1),
        .gen_image_0 (gen_image_0),
        .gen_image_1 (gen_image_1),
        .gen_image_2 (gen_image_2),
        .gen_image_3 (gen_image_3),
        .gen_image_4 (gen_image_4),
        .gen_image_5 (gen_image_5),
        .gen_image_6 (gen_image_6),
        .gen_image_7 (gen_image_7),
        .gen_image_8 (gen_image_8),
        .disc_prob   (disc_prob),
        .done        (done),
        .gen_valid   (gen_valid),
        .disc_valid  (disc_valid)
    );

    // ========================================================================
    // Sample Input Memory
    // ========================================================================
    // Load sample inputs (if available)
    reg signed [15:0] sample_inputs [0:19]; // 10 samples x 2 values
    integer sample_file_exists;
    
    initial begin
        sample_file_exists = 0;
        // Try to load sample inputs
        $readmemh("parameters/input_sample_00_q15.hex", sample_inputs);
        if (sample_inputs[0] !== 16'hxxxx) begin
            sample_file_exists = 1;
            $display("[INFO] Loaded sample inputs from input_sample_00_q15.hex");
        end else begin
            $display("[WARN] Sample input file not found, using random values");
        end
    end

    // ========================================================================
    // Utility Functions
    // ========================================================================
    // Convert Q1.15 to real for display
    function real q15_to_real;
        input signed [15:0] q15_val;
        begin
            q15_to_real = $itor(q15_val) / 32768.0;
        end
    endfunction
    
    // Convert real to Q1.15
    function signed [15:0] real_to_q15;
        input real r_val;
        begin
            real_to_q15 = $rtoi(r_val * 32768.0);
        end
    endfunction

    // ========================================================================
    // Display Tasks
    // ========================================================================
    task display_generator_output;
        integer row, col, idx;
        real pixel_val;
        begin
            $display("\n========================================");
            $display("GENERATOR OUTPUT (3x3 Image)");
            $display("========================================");
            
            // Display as matrix
            for (row = 0; row < 3; row = row + 1) begin
                $write("  [");
                for (col = 0; col < 3; col = col + 1) begin
                    idx = row * 3 + col;
                    case (idx)
                        0: pixel_val = q15_to_real(gen_image_0);
                        1: pixel_val = q15_to_real(gen_image_1);
                        2: pixel_val = q15_to_real(gen_image_2);
                        3: pixel_val = q15_to_real(gen_image_3);
                        4: pixel_val = q15_to_real(gen_image_4);
                        5: pixel_val = q15_to_real(gen_image_5);
                        6: pixel_val = q15_to_real(gen_image_6);
                        7: pixel_val = q15_to_real(gen_image_7);
                        8: pixel_val = q15_to_real(gen_image_8);
                        default: pixel_val = 0.0;
                    endcase
                    
                    if (col < 2)
                        $write("%7.4f, ", pixel_val);
                    else
                        $write("%7.4f", pixel_val);
                end
                $display("]");
            end
            
            // Display raw hex values
            $display("\nRaw Hex Values:");
            $display("  [%04h %04h %04h]", gen_image_0, gen_image_1, gen_image_2);
            $display("  [%04h %04h %04h]", gen_image_3, gen_image_4, gen_image_5);
            $display("  [%04h %04h %04h]", gen_image_6, gen_image_7, gen_image_8);
        end
    endtask
    
    task display_discriminator_output;
        real prob_real;
        begin
            prob_real = q15_to_real(disc_prob);
            $display("\n========================================");
            $display("DISCRIMINATOR OUTPUT");
            $display("========================================");
            $display("  Probability (Real): %7.4f", prob_real);
            $display("  Raw Hex Value:      0x%04h", disc_prob);
            
            if (prob_real > 0.5)
                $display("  Classification:     REAL (confidence: %.1f%%)", prob_real * 100.0);
            else
                $display("  Classification:     FAKE (confidence: %.1f%%)", (1.0 - prob_real) * 100.0);
        end
    endtask

    // ========================================================================
    // Test Task
    // ========================================================================
    task run_test;
        input signed [15:0] n0, n1;
        input [80*8-1:0] test_name;
        begin
            $display("\n");
            $display("========================================");
            $display("TEST: %0s", test_name);
            $display("========================================");
            $display("Input Noise: [%7.4f, %7.4f]", q15_to_real(n0), q15_to_real(n1));
            $display("         Hex: [0x%04h, 0x%04h]", n0, n1);
            
            // Apply inputs
            noise_0 = n0;
            noise_1 = n1;
            start = 1'b1;
            
            @(posedge clk);
            start = 1'b0;
            
            // Wait for done
            wait(done);
            @(posedge clk);
            
            // Display results
            display_generator_output();
            display_discriminator_output();
            
            // Wait a bit before next test
            repeat(3) @(posedge clk);
        end
    endtask

    // ========================================================================
    // Main Test Sequence
    // ========================================================================
    initial begin
        // Initialize VCD dump
        $dumpfile("simple_gan_tb.vcd");
        $dumpvars(0, simple_gan_tb);
        
        // Initialize signals
        rst_n = 0;
        start = 0;
        noise_0 = 16'h0;
        noise_1 = 16'h0;
        
        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        $display("\n");
        $display("========================================");
        $display("Simple-GAN 3x3 Testbench");
        $display("========================================");
        $display("Clock Period:    %0d ns", CLK_PERIOD);
        $display("Start Time:      %0t", $time);
        
        // Test 1: Zero input
        run_test(16'h0000, 16'h0000, "Zero Input");
        
        // Test 2: Positive noise
        run_test(real_to_q15(0.5), real_to_q15(0.5), "Positive Noise [0.5, 0.5]");
        
        // Test 3: Negative noise
        run_test(real_to_q15(-0.5), real_to_q15(-0.5), "Negative Noise [-0.5, -0.5]");
        
        // Test 4: Mixed noise
        run_test(real_to_q15(0.7), real_to_q15(-0.3), "Mixed Noise [0.7, -0.3]");
        
        // Test 5: Large positive
        run_test(real_to_q15(0.9), real_to_q15(0.8), "Large Positive [0.9, 0.8]");
        
        // Test 6: From sample file (if available)
        if (sample_file_exists) begin
            run_test(sample_inputs[0], sample_inputs[1], "Sample File Input #0");
        end
        
        // Test 7: Random inputs
        noise_0 = $random;
        noise_1 = $random;
        run_test(noise_0, noise_1, "Random Input");
        
        // End simulation
        $display("\n");
        $display("========================================");
        $display("SIMULATION COMPLETE");
        $display("========================================");
        $display("End Time:        %0t", $time);
        $display("Total Duration:  %0t ns", $time);
        $display("\nTo view waveforms: gtkwave simple_gan_tb.vcd");
        $display("\n");
        
        repeat(10) @(posedge clk);
        $finish;
    end
    
    // ========================================================================
    // Timeout Watchdog
    // ========================================================================
    initial begin
        #100000; // 100us timeout
        $display("\n[ERROR] Simulation timeout!");
        $finish;
    end

endmodule
