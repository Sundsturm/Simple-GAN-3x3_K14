#!/bin/bash
# Simple-GAN 3x3 Simulation Script
# This script converts parameters to hex and runs the testbench

echo "========================================"
echo "Simple-GAN 3x3 Simulation"
echo "========================================"

# Step 1: Convert parameters to hex
echo ""
echo "[Step 1] Converting parameters to Q1.15 hex format..."
python scripts/convert_params_to_hex.py

# Step 2: Compile with Icarus Verilog
echo ""
echo "[Step 2] Compiling Verilog files..."
iverilog -o simple_gan_tb_out \
    rtl/simple_gan_top.v \
    rtl/generator_q15.v \
    rtl/discriminator_q15.v \
    rtl/activation_unit/gau_q15.v \
    rtl/activation_unit/glu_activation.v \
    rtl/activation_unit/leaky_relu_q15.v \
    rtl/activation_unit/pwl_activation.v \
    rtl/activation_unit/pwl_sigmoid.v \
    rtl/activation_unit/sigmoid_approx_q15.v \
    rtl/activation_unit/tanh_approx_q15.v \
    tb/simple_gan_tb.v

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"

# Step 3: Run simulation
echo ""
echo "[Step 3] Running simulation..."
vvp simple_gan_tb_out

echo ""
echo "========================================"
echo "Simulation Complete!"
echo "Waveform file: simple_gan_tb.vcd"
echo "To view waveforms: gtkwave simple_gan_tb.vcd"
echo "========================================"
