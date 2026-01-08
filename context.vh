// === CONTEXT FOR COPILOT - Simple-GAN 3x3 (O+ Pattern Generator)
// You are helping write RTL Verilog (NOT SystemVerilog) for a Simple-GAN
// hardware implementation that generates 3x3 images (circle and cross patterns).
// Training reference: LSI_Contest_simple_gan_3x3.m in matlab/ folder

// GENERAL CONSTRAINTS:
// - Use pure Verilog-2001 syntax only.
// - Target tool: Icarus Verilog (IVerilog) and GTKWave for simulation.
// - No vendor-specific IP, no AXI, no DMA, no FPGA primitives.
// - Simulation-first design (IVerilog compatible).
// - Fixed-point arithmetic only: Q1.15 format (16-bit signed).
// - Weights and inputs are loaded from .txt files using $readmemh.
// - All memories are modeled as reg arrays.
// - No floating-point, no runtime quantizer module.
// - No SystemVerilog constructs (no logic type, no always_comb, etc.).

// ARCHITECTURE OVERVIEW:
// - The design implements ONLY inference (forward pass).
// - Two main networks: Generator (G) and Discriminator (D).
// - Network dimensions:
//   * Latent dimension: 2 (input noise)
//   * Hidden layer size: 3 neurons
//   * Output size: 9 (3x3 image)
//
// GENERATOR G:
//   Input: noise vector (2D) -> Hidden: 3 neurons -> Output: 9 pixels
//   Layer 2: FC(2->3) + tanh
//   Layer 3: FC(3->9) + tanh
//   Parameters: Wg2[3x2], bg2[3], Wg3[9x3], bg3[9]
//
// DISCRIMINATOR D:
//   Input: image (9D) -> Hidden: 3 neurons -> Output: 1 (real/fake)
//   Layer 2: FC(9->3) + tanh
//   Layer 3: FC(3->1) + sigmoid
//   Parameters: Wd2[3x9], bd2[3], Wd3[1x3], bd3[1]

// CORE MODULES:
// - neuron: single neuron (MAC + bias + activation).
// - fc_layer: fully-connected layer with N neurons.
// - mac_unit: multiply-accumulate with Q1.15 fixed-point saturation.
// - tanh_approx_q15: piecewise-linear tanh approximation.
// - sigmoid_approx_q15: piecewise-linear sigmoid approximation.
// - weight_mem / bias_mem: reg-array-based memory initialized via $readmemh.

// GENERATOR G MODULES:
// - generator_layer2: FC(2->3) + tanh
// - generator_layer3: FC(3->9) + tanh
// - generator_top: structural wrapper for Generator G.
// - generator_fsm: FSM sequencing Generator layers (L2 -> L3 -> done).

// DISCRIMINATOR D MODULES:
// - discriminator_layer2: FC(9->3) + tanh
// - discriminator_layer3: FC(3->1) + sigmoid
// - discriminator_top: structural wrapper for Discriminator D.
// - discriminator_fsm: FSM sequencing Discriminator layers (L2 -> L3 -> done).

// DESIGN STYLE RULES:
// - Separate datapath and control clearly.
// - Use start/done handshake signals between modules.
// - Avoid monolithic always blocks; prefer readable FSMs.
// - Use parameters for bit width and layer dimensions.
// - Comment each module header with:
//   - Purpose
//   - Inputs/outputs
//   - Fixed-point assumptions (Q1.15 format)
// - All arithmetic must handle Q1.15 fixed-point:
//   * Multiplication: (a * b) >> 15 to normalize
//   * Addition/Subtraction: direct with saturation check
//   * Range: -1.0 (0x8000) to +0.999969 (0x7FFF)

// MEMORY RULES:
// - .txt files contain hex values (Q1.15 format).
// - Use $readmemh to initialize weight/bias memories.
// - Memory must be explicitly declared as reg arrays in Verilog.
// - Weights are pre-quantized offline (see extract_gan_parameters.py).
// - Memory initialization example:
//   reg signed [15:0] weight_mem [0:5]; // 6 weights
//   initial $readmemh("Wg2.txt", weight_mem);

// FSM RULES:
// - Use explicit state encoding with localparam.
// - One always block for state register (sequential logic).
// - One always block for next-state logic (combinational).
// - One always block for output/control signals (combinational).
// - State naming: IDLE, LOAD_INPUTS, COMPUTE_L2, COMPUTE_L3, DONE
// - Use start/done handshake protocol:
//   * External module asserts 'start' signal
//   * FSM asserts 'done' when computation finished
//   * FSM returns to IDLE after 'done' acknowledged

// FIXED-POINT ARITHMETIC (Q1.15):
// - Format: 1 sign bit + 15 fractional bits
// - Range: -1.0 to +0.999969
// - Multiplication: result = (a * b) >>> 15 (arithmetic right shift)
// - Saturation: clamp result to [0x8000, 0x7FFF]
// - Example MAC operation:
//   wire signed [31:0] prod = w * x;
//   wire signed [15:0] prod_q15 = prod[30:15]; // extract Q1.15
//   wire signed [16:0] sum = acc + prod_q15;   // with overflow bit
//   wire signed [15:0] result = (sum[16]) ? 
//                                (sum[15] ? 16'h8000 : 16'h7FFF) : sum[15:0];

// ACTIVATION FUNCTIONS:
// - tanh_approx_q15: piecewise-linear approximation
//   * |x| < 0.5: y = x (linear region)
//   * 0.5 <= |x| < 0.9: y = ±0.8
//   * |x| >= 0.9: y = ±0.99
// - sigmoid_approx_q15: piecewise-linear approximation
//   * x <= -0.5: y = 0.25
//   * x >= +0.5: y = 0.75
//   * else: y = 0.5 + x/2

// OUTPUT EXPECTATION:
// - Generate clean, synthesizable Verilog-2001.
// - Prefer clarity over extreme optimization.
// - Assume correctness and debuggability are more important than speed.
// - All modules must be IVerilog-compatible (no vendor-specific primitives).
// - Include testbench template with $readmemh for weight loading.

// When generating code:
// - Only generate the module requested.
// - Do not invent AXI, DMA, or FPGA-specific interfaces.
// - Do not use SystemVerilog features (logic, always_comb, always_ff, etc.).
// - Use standard Verilog-2001 constructs only:
//   * module, endmodule
//   * input, output, inout
//   * wire, reg
//   * always @(*), always @(posedge clk)
//   * assign
//   * localparam, parameter
// - Avoid non-blocking assignments (<=) in combinational logic.
// - Use blocking assignments (=) in combinational always blocks.
// - Use non-blocking assignments (<=) in sequential always blocks.
// =======================================================================
