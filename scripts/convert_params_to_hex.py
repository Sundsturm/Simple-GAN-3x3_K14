#!/usr/bin/env python3
"""
Convert decimal floating-point parameter files to Q1.15 hexadecimal format.
Q1.15 format: 1 sign bit, 15 fractional bits
Range: -1.0 to +0.999969482421875
"""

import os
import struct

def float_to_q15_hex(value):
    """Convert a float to Q1.15 fixed-point hex string."""
    # Clamp value to Q1.15 range
    value = max(-1.0, min(value, 0.999969482421875))
    
    # Convert to Q1.15: multiply by 2^15 and round
    q15_val = int(round(value * 32768))
    
    # Handle overflow
    if q15_val > 32767:
        q15_val = 32767
    elif q15_val < -32768:
        q15_val = -32768
    
    # Convert to unsigned 16-bit for hex representation
    if q15_val < 0:
        q15_val = q15_val + 65536  # Two's complement
    
    return format(q15_val, '04X')

def convert_file(input_path, output_path):
    """Convert a parameter file from decimal to hex format."""
    values = []
    
    with open(input_path, 'r') as f:
        for line in f:
            # Split line by whitespace and parse each number
            for token in line.strip().split():
                if token:
                    try:
                        val = float(token)
                        values.append(val)
                    except ValueError:
                        print(f"Warning: Could not parse '{token}' in {input_path}")
    
    # Write hex values, one per line
    with open(output_path, 'w') as f:
        for val in values:
            hex_val = float_to_q15_hex(val)
            f.write(hex_val + '\n')
    
    print(f"Converted {input_path} -> {output_path} ({len(values)} values)")

def main():
    # Get the parameters directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_dir = os.path.join(script_dir, '..', 'parameters')
    hex_dir = os.path.join(params_dir, 'hex')
    
    # Create hex output directory
    os.makedirs(hex_dir, exist_ok=True)
    
    # Parameter files to convert
    param_files = [
        'Wg2.txt', 'bg2.txt', 'Wg3.txt', 'bg3.txt',  # Generator
        'Wd2.txt', 'bd2.txt', 'Wd3.txt', 'bd3.txt',  # Discriminator
    ]
    
    # Also convert input sample files
    for i in range(10):
        param_files.append(f'input_sample_{i:02d}.txt')
    
    for filename in param_files:
        input_path = os.path.join(params_dir, filename)
        output_path = os.path.join(hex_dir, filename)
        
        if os.path.exists(input_path):
            convert_file(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found, skipping")

if __name__ == '__main__':
    main()
