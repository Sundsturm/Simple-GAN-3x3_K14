%% =========================================================
%  Extract Simple-GAN 3x3 Parameters
%  Extracts trained weights, biases, and generates sample inputs
% =========================================================
clear; clc;
close all;

fprintf("="*ones(1,60) + "\n");
fprintf("Simple-GAN 3x3 Parameter Extraction\n");
fprintf("="*ones(1,60) + "\n\n");

%% Load trained model
mat_file = 'trained_simple_gan.mat';
if ~isfile(mat_file)
    error('File %s not found! Please run training script first.', mat_file);
end

fprintf("Loading trained model from %s...\n", mat_file);
load(mat_file);
fprintf("✓ Model loaded successfully\n");

%% Display Generator parameters
fprintf("\n" + "="*ones(1,60) + "\n");
fprintf("GENERATOR PARAMETERS\n");
fprintf("="*ones(1,60) + "\n");

fprintf("\nWg2 (3x2) - First layer weights:\n");
disp(Wg2);

fprintf("bg2 (3x1) - First layer bias:\n");
disp(bg2');

fprintf("\nWg3 (9x3) - Second layer weights:\n");
disp(Wg3);

fprintf("bg3 (9x1) - Second layer bias:\n");
disp(bg3');

%% Display Discriminator parameters
fprintf("\n" + "="*ones(1,60) + "\n");
fprintf("DISCRIMINATOR PARAMETERS\n");
fprintf("="*ones(1,60) + "\n");

fprintf("\nWd2 (3x9) - First layer weights:\n");
disp(Wd2);

fprintf("bd2 (3x1) - First layer bias:\n");
disp(bd2');

fprintf("\nWd3 (1x3) - Second layer weights:\n");
disp(Wd3);

fprintf("bd3 (scalar) - Second layer bias:\n");
disp(bd3);

%% Generate sample inputs
num_samples = 10;
latent_dim = 2;
rng(42);  % Set seed for reproducibility

fprintf("\n✓ Generating %d sample inputs...\n", num_samples);
sample_inputs = cell(num_samples, 1);
for i = 1:num_samples
    sample_inputs{i} = randn(latent_dim, 1);
end

%% Save parameters to text files
fprintf("\n=== Saving parameters to text files ===\n");

% Generator parameters
writematrix(Wg2, 'Wg2.txt', 'Delimiter', ' ');
writematrix(bg2, 'bg2.txt', 'Delimiter', ' ');
writematrix(Wg3, 'Wg3.txt', 'Delimiter', ' ');
writematrix(bg3, 'bg3.txt', 'Delimiter', ' ');

% Discriminator parameters
writematrix(Wd2, 'Wd2.txt', 'Delimiter', ' ');
writematrix(bd2, 'bd2.txt', 'Delimiter', ' ');
writematrix(Wd3, 'Wd3.txt', 'Delimiter', ' ');
writematrix(bd3, 'bd3.txt', 'Delimiter', ' ');

fprintf("✓ Parameters saved to text files\n");

%% Convert to Q1.15 fixed-point format
fprintf("\n=== Converting to Q1.15 fixed-point format ===\n");

function q15_val = to_q15(value)
    % Convert floating point to Q1.15 (16-bit signed)
    % Range: -1.0 to 0.999969
    value = max(min(value, 0.999969), -1.0);  % Clamp
    q15_val = round(value * 32768);  % Scale to 2^15
    q15_val = int16(q15_val);
end

function save_q15_hex(arr, filename)
    % Save array in Q1.15 hex format
    arr_flat = arr(:);  % Flatten
    fid = fopen(filename, 'w');
    for i = 1:length(arr_flat)
        q15_val = to_q15(arr_flat(i));
        % Convert to unsigned 16-bit hex
        hex_val = typecast(q15_val, 'uint16');
        fprintf(fid, '%04X\n', hex_val);
    end
    fclose(fid);
end

% Save Generator parameters in Q1.15 hex
save_q15_hex(Wg2, 'Wg2_q15.hex');
save_q15_hex(bg2, 'bg2_q15.hex');
save_q15_hex(Wg3, 'Wg3_q15.hex');
save_q15_hex(bg3, 'bg3_q15.hex');

% Save Discriminator parameters in Q1.15 hex
save_q15_hex(Wd2, 'Wd2_q15.hex');
save_q15_hex(bd2, 'bd2_q15.hex');
save_q15_hex(Wd3, 'Wd3_q15.hex');
save_q15_hex(bd3, 'bd3_q15.hex');

fprintf("✓ Q1.15 hex parameters saved\n");

%% Save sample inputs
fprintf("\n=== Saving %d sample inputs ===\n", num_samples);

for i = 1:num_samples
    % Save as text
    filename_txt = sprintf('input_sample_%02d.txt', i-1);
    writematrix(sample_inputs{i}, filename_txt, 'Delimiter', ' ');
    
    % Save as Q1.15 hex
    filename_hex = sprintf('input_sample_%02d_q15.hex', i-1);
    save_q15_hex(sample_inputs{i}, filename_hex);
end

fprintf("✓ Sample inputs saved\n");

%% Save all parameters in a structured MAT file
export_data = struct();
export_data.generator.Wg2 = Wg2;
export_data.generator.bg2 = bg2;
export_data.generator.Wg3 = Wg3;
export_data.generator.bg3 = bg3;
export_data.discriminator.Wd2 = Wd2;
export_data.discriminator.bd2 = bd2;
export_data.discriminator.Wd3 = Wd3;
export_data.discriminator.bd3 = bd3;
export_data.sample_inputs = sample_inputs;

save('gan_parameters_extracted.mat', '-struct', 'export_data');
fprintf("\n✓ Structured parameters saved to gan_parameters_extracted.mat\n");

%% Generate and visualize samples
fprintf("\n=== Generating sample outputs ===\n");

tanh_f = @(x) tanh(x);

figure('Name', 'Generated Samples from Extracted Parameters');
for i = 1:9
    ng = sample_inputs{i};
    
    % Forward pass through Generator
    ag2 = tanh_f(Wg2*ng + bg2);
    x_fake = tanh_f(Wg3*ag2 + bg3);
    
    % Reshape and display
    img = reshape(x_fake/2 + 0.5, 3, 3);
    
    subplot(3, 3, i);
    imagesc(img);
    colormap gray;
    axis image off;
    title(sprintf('Sample %d', i));
end
sgtitle('Generated Images from Extracted Model');

% Save figure
exportgraphics(gcf, 'extracted_samples.png');
fprintf("✓ Sample images saved to extracted_samples.png\n");

%% Summary
fprintf("\n" + "="*ones(1,60) + "\n");
fprintf("EXTRACTION COMPLETE\n");
fprintf("="*ones(1,60) + "\n");
fprintf("\nGenerated files:\n");
fprintf("  Text format:\n");
fprintf("    - Wg2.txt, bg2.txt, Wg3.txt, bg3.txt\n");
fprintf("    - Wd2.txt, bd2.txt, Wd3.txt, bd3.txt\n");
fprintf("  Q1.15 hex format (for RTL):\n");
fprintf("    - *_q15.hex files\n");
fprintf("  Sample inputs:\n");
fprintf("    - input_sample_*.txt\n");
fprintf("    - input_sample_*_q15.hex\n");
fprintf("  MAT file:\n");
fprintf("    - gan_parameters_extracted.mat\n");
fprintf("  Images:\n");
fprintf("    - extracted_samples.png\n");
fprintf("\n✓ Ready for hardware implementation!\n");
