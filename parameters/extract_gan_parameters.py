"""
Extract Simple-GAN 3x3 Parameters
Extracts trained weights, biases, and generates sample inputs from trained GAN model
"""

import numpy as np
from scipy.io import loadmat, savemat
import json

def load_trained_model(mat_file='trained_simple_gan.mat'):
    """Load trained GAN parameters from .mat file"""
    print(f"Loading trained model from {mat_file}...")
    data = loadmat(mat_file)
    
    # Extract Generator parameters
    Wg2 = data['Wg2']  # Shape: (G_hidden_L, latent_dim) = (3, 2)
    bg2 = data['bg2']  # Shape: (G_hidden_L, 1) = (3, 1)
    Wg3 = data['Wg3']  # Shape: (img_size^2, G_hidden_L) = (9, 3)
    bg3 = data['bg3']  # Shape: (img_size^2, 1) = (9, 1)
    
    # Extract Discriminator parameters
    Wd2 = data['Wd2']  # Shape: (D_hidden_L, img_size^2) = (3, 9)
    bd2 = data['bd2']  # Shape: (D_hidden_L, 1) = (3, 1)
    Wd3 = data['Wd3']  # Shape: (1, D_hidden_L) = (1, 3)
    bd3 = data['bd3']  # Shape: scalar
    
    return {
        'generator': {'Wg2': Wg2, 'bg2': bg2, 'Wg3': Wg3, 'bg3': bg3},
        'discriminator': {'Wd2': Wd2, 'bd2': bd2, 'Wd3': Wd3, 'bd3': bd3}
    }

def generate_sample_inputs(num_samples=10, latent_dim=2, seed=42):
    """Generate sample latent vectors (noise inputs) for Generator"""
    np.random.seed(seed)
    samples = []
    for i in range(num_samples):
        noise = np.random.randn(latent_dim, 1)
        samples.append(noise)
    return samples

def save_parameters_to_txt(params, output_dir='./'):
    """Save all parameters to text files (useful for RTL testbench)"""
    print("\n=== Saving parameters to text files ===")
    
    # Generator parameters
    np.savetxt(f'{output_dir}Wg2.txt', params['generator']['Wg2'], fmt='%.8f')
    np.savetxt(f'{output_dir}bg2.txt', params['generator']['bg2'], fmt='%.8f')
    np.savetxt(f'{output_dir}Wg3.txt', params['generator']['Wg3'], fmt='%.8f')
    np.savetxt(f'{output_dir}bg3.txt', params['generator']['bg3'], fmt='%.8f')
    
    # Discriminator parameters
    np.savetxt(f'{output_dir}Wd2.txt', params['discriminator']['Wd2'], fmt='%.8f')
    np.savetxt(f'{output_dir}bd2.txt', params['discriminator']['bd2'], fmt='%.8f')
    np.savetxt(f'{output_dir}Wd3.txt', params['discriminator']['Wd3'], fmt='%.8f')
    np.savetxt(f'{output_dir}bd3.txt', params['discriminator']['bd3'].reshape(1, -1), fmt='%.8f')
    
    print(f"Parameters saved to {output_dir}")

def save_parameters_to_json(params, filename='gan_parameters.json'):
    """Save parameters to JSON format"""
    # Convert numpy arrays to lists
    params_json = {
        'generator': {
            'Wg2': params['generator']['Wg2'].tolist(),
            'bg2': params['generator']['bg2'].tolist(),
            'Wg3': params['generator']['Wg3'].tolist(),
            'bg3': params['generator']['bg3'].tolist()
        },
        'discriminator': {
            'Wd2': params['discriminator']['Wd2'].tolist(),
            'bd2': params['discriminator']['bd2'].tolist(),
            'Wd3': params['discriminator']['Wd3'].tolist(),
            'bd3': params['discriminator']['bd3'].tolist()
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(params_json, f, indent=2)
    print(f"Parameters saved to {filename}")

def convert_to_fixed_point_q15(value):
    """Convert floating point to Q1.15 fixed-point format (16-bit)"""
    # Clamp to [-1.0, 0.999969]
    value = np.clip(value, -1.0, 0.999969)
    # Scale to Q1.15: multiply by 2^15 = 32768
    q15_value = np.round(value * 32768).astype(np.int16)
    return q15_value

def save_parameters_to_q15_hex(params, output_dir='./'):
    """Save parameters in Q1.15 fixed-point hex format (for RTL)"""
    print("\n=== Saving parameters in Q1.15 hex format ===")
    
    def save_q15_hex(arr, filename):
        arr_flat = arr.flatten()
        q15_values = convert_to_fixed_point_q15(arr_flat)
        with open(filename, 'w') as f:
            for val in q15_values:
                # Convert to 16-bit hex (handle negative numbers properly)
                hex_val = int(val) & 0xFFFF
                f.write(f"{hex_val:04X}\n")
    
    # Save Generator parameters
    save_q15_hex(params['generator']['Wg2'], f'{output_dir}Wg2_q15.hex')
    save_q15_hex(params['generator']['bg2'], f'{output_dir}bg2_q15.hex')
    save_q15_hex(params['generator']['Wg3'], f'{output_dir}Wg3_q15.hex')
    save_q15_hex(params['generator']['bg3'], f'{output_dir}bg3_q15.hex')
    
    # Save Discriminator parameters
    save_q15_hex(params['discriminator']['Wd2'], f'{output_dir}Wd2_q15.hex')
    save_q15_hex(params['discriminator']['bd2'], f'{output_dir}bd2_q15.hex')
    save_q15_hex(params['discriminator']['Wd3'], f'{output_dir}Wd3_q15.hex')
    save_q15_hex(params['discriminator']['bd3'], f'{output_dir}bd3_q15.hex')
    
    print(f"Q1.15 hex parameters saved to {output_dir}")

def save_sample_inputs(samples, output_dir='./'):
    """Save sample input vectors"""
    print(f"\n=== Saving {len(samples)} sample inputs ===")
    
    # Save as text
    for i, sample in enumerate(samples):
        np.savetxt(f'{output_dir}input_sample_{i:02d}.txt', sample, fmt='%.8f')
    
    # Save as Q1.15 hex
    for i, sample in enumerate(samples):
        q15_sample = convert_to_fixed_point_q15(sample)
        with open(f'{output_dir}input_sample_{i:02d}_q15.hex', 'w') as f:
            for val in q15_sample.flatten():
                hex_val = int(val) & 0xFFFF
                f.write(f"{hex_val:04X}\n")
    
    print(f"Sample inputs saved to {output_dir}")

def display_parameters(params):
    """Display parameters in readable format"""
    print("\n" + "="*60)
    print("GENERATOR PARAMETERS")
    print("="*60)
    print("\nWg2 (3x2) - First layer weights:")
    print(params['generator']['Wg2'])
    print("\nbg2 (3x1) - First layer bias:")
    print(params['generator']['bg2'].flatten())
    print("\nWg3 (9x3) - Second layer weights:")
    print(params['generator']['Wg3'])
    print("\nbg3 (9x1) - Second layer bias:")
    print(params['generator']['bg3'].flatten())
    
    print("\n" + "="*60)
    print("DISCRIMINATOR PARAMETERS")
    print("="*60)
    print("\nWd2 (3x9) - First layer weights:")
    print(params['discriminator']['Wd2'])
    print("\nbd2 (3x1) - First layer bias:")
    print(params['discriminator']['bd2'].flatten())
    print("\nWd3 (1x3) - Second layer weights:")
    print(params['discriminator']['Wd3'])
    print("\nbd3 (scalar) - Second layer bias:")
    print(params['discriminator']['bd3'])

def main():
    """Main extraction function"""
    print("="*60)
    print("Simple-GAN 3x3 Parameter Extraction")
    print("="*60)
    
    # Load trained model
    try:
        params = load_trained_model('trained_simple_gan.mat')
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("✗ Error: 'trained_simple_gan.mat' not found!")
        print("  Please run the MATLAB training script first.")
        return
    
    # Display parameters
    display_parameters(params)
    
    # Generate sample inputs
    num_samples = 10
    samples = generate_sample_inputs(num_samples=num_samples, latent_dim=2, seed=42)
    print(f"\n✓ Generated {num_samples} sample inputs")
    
    # Save in different formats
    save_parameters_to_txt(params, output_dir='./')
    save_parameters_to_json(params, filename='gan_parameters.json')
    save_parameters_to_q15_hex(params, output_dir='./')
    save_sample_inputs(samples, output_dir='./')
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  Text format: Wg2.txt, bg2.txt, Wg3.txt, bg3.txt, Wd2.txt, bd2.txt, Wd3.txt, bd3.txt")
    print("  JSON format: gan_parameters.json")
    print("  Q1.15 hex:   *_q15.hex files (for RTL testbench)")
    print("  Inputs:      input_sample_*.txt and input_sample_*_q15.hex")
    print("\nReady for hardware implementation!")

if __name__ == '__main__':
    main()
