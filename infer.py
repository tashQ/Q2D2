# --coding:utf-8--
import os
import shutil
from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer
import numpy as np
from collections import defaultdict
import time
import logging
from scipy.stats import entropy


def decode_to_pairs(flat_idx, grid_lens):
    """
    Convert flattened composite indices into per-pair indices.

    flat_idx: 1D tensor of shape [M], each entry is a flattened code index
    grid_lens: list or numpy array with size of each pair grid [G1, G2, ..., GP]
    returns: tensor [M, P] with per-pair indices
    """
    device = flat_idx.device
    grid_lens = torch.tensor(grid_lens, device=device, dtype=torch.long)
    P = len(grid_lens)

    # compute bases like [1, G1, G1*G2, ...]
    bases = torch.ones(P, dtype=torch.long, device=device)
    for i in range(1, P):
        bases[i] = bases[i-1] * grid_lens[i-1]

    pair_idx = []
    for i in range(P):
        idx_i = (flat_idx // bases[i]) % grid_lens[i]
        pair_idx.append(idx_i)

    return torch.stack(pair_idx, dim=-1)  # [M, P]

##################configs####################################
device1=torch.device('cuda:0')
# device2=torch.device('cpu')

out_folder = "path/to/infer/dir"
ll="new_folder_name"
config_path = "path/to/dir/config.yaml"
model_path = "path/to/dir/checkpoints"
ckpt_name = "xxxx.ckpt"


#libritts_test_clean, libritts_test_other, LJSpeech, librispeech_test_clean
dataset = "libritts_test_clean"

#log dir - dont touch
log_dir = (out_folder + "/" + ll)


if dataset == "libritts_test_clean":
    dataset_dir = "path/to/LibriTTS/test-clean/"
    ll = (ll + "/libritts_test_clean")
elif dataset == "libritts_test_other":
    dataset_dir = "path/to/LibriTTS/test-other/"
    ll = (ll + "/libritts_test_other")
elif dataset == "LJSpeech":
    dataset_dir = "path/to/LJSpeech-1.1/wavs"
    ll = (ll + "/LJSpeech")
elif dataset == "librispeech_test_clean":
    dataset_dir = "path/to/LibriSpeech/test-clean"
    ll = (ll + "/librispeech_test_clean")

# Paths for input and output
input_dir = (out_folder + "/" + ll + "/in")
output_dir = (out_folder + "/" + ll + "/out")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ---Save checkpoint in log_dir if not already there ---
ckpt_target = os.path.join(log_dir, os.path.basename(ckpt_name))
model_path = model_path + "/" + ckpt_name
if not os.path.exists(ckpt_target):
    shutil.copy2(model_path, ckpt_target)
    print(f"Checkpoint copied to {ckpt_target}")
else:
    print(f"Checkpoint already exists at {ckpt_target}, skipping copy.")
    



#for utilization
# user-provided grid config
grid_type = "rhombic"   # "hex" or "cube", "rhombic", ...
levels = [7, 7, 7, 7, 7, 7]   # example, user fills in
num_pairs = len(levels) // 2

# lookup for grid sizes
GRID_SIZES = {
    "hex": {3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81, 10: 100, 11: 121, 12: 144, 13: 169},
    "rhombic": {3: 5, 4: 13, 5: 25, 6: 41, 7: 61, 8: 85, 9: 113, 10: 145, 11: 181, 12: 221, 13: 265},
    "cube": {3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81, 10: 100, 11: 121, 12: 144, 13: 169}
}

grid_lens = []
for p in range(num_pairs):
    L = levels[2*p]   # level for this pair
    grid_lens.append(GRID_SIZES[grid_type][L])
grid_lens = np.array(grid_lens)


# === Save run information ===
info_file = os.path.join(log_dir, "experiment_info.txt")
with open(info_file, "w", encoding="utf-8") as f:
    f.write("Experiment Information\n")
    f.write("======================\n")
    f.write(f"Config file: {config_path}\n")
    f.write(f"Checkpoint: {model_path}\n")
print(f"Saved experiment info to {info_file}")

print(f"model: {model_path}")
print(f"output dir: {output_dir}")


###################wavtok########################
q2d2 = WavTokenizer.from_pretrained0802(config_path, model_path)
q2d2 = wavtokenizer.to(device1)


# Track progress
total_files = 0
processed_files = 0
skipped_files = 0

features_all=[]
total_dur = 0
total_proc = 0

# trackers
pair_indices = [set() for _ in range(num_pairs)]
composite_indices = set()

# Process all .wav files
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith((".wav", ".flac")):
            total_files += 1
            wav_path = os.path.join(root, file)
            name, _ = os.path.splitext(file)   # strip extension safely
            input_save_path = os.path.join(input_dir, f"{name}.in.wav")
            output_save_path = os.path.join(output_dir, f"{name}.out.wav")


            # Skip if output file already exists
            if os.path.exists(output_save_path):
                print(f"Skipping {file} (already processed)")
                skipped_files += 1
                continue

            # Load and process audio
            wav, sr = torchaudio.load(wav_path)
            wav = convert_audio(wav, sr, 24000, 1)
            duration_sec = wav.size(1) / sr  # wav.size(1) = number of samples, sr = sample rate
            bandwidth_id = torch.tensor([0])
            wav=wav.to(device1)


            #start time
            start = time.time()

            features, discrete_code = q2d2.encode_infer(wav, bandwidth_id=bandwidth_id)
            
            #utilization
            flat = discrete_code.view(-1)  # [B*N]
            arr = decode_to_pairs(flat, grid_lens).cpu().numpy()
            for row in arr:
                for p, idx in enumerate(row):
                    pair_indices[p].add(int(idx))
                composite_indices.add(tuple(row))
            
            # Fix dimension order: [B, T, C] -> [B, C, T]
            if features.dim() == 3 and features.shape[-1] == 512:  # last dim is feature dim
                features = features.permute(0, 2, 1).contiguous()

            features_all.append(features)

            bandwidth_id = bandwidth_id.to(device1)
            audio_out = q2d2.decode(features, bandwidth_id=bandwidth_id)
            
            #end time
            end = time.time()

            
            # Save original and processed audio
            torchaudio.save(input_save_path, wav.cpu(), sample_rate=24000)
            torchaudio.save(output_save_path, audio_out.cpu() ,sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
            processed_files += 1
            
            #rtf calculation
            processing_time = end - start
            rtf = processing_time / duration_sec  # real-time factor
            utilization = 100 / rtf if rtf > 0 else 0
            print(f"{file}: duration={duration_sec:.2f}s, proc_time={processing_time:.2f}s, RTF={rtf:.3f}, Utilization={utilization:.1f}%")
            total_dur += duration_sec
            total_proc += processing_time
            
# Summary
print("\nProcessing complete!")
print(f"Total files found: {total_files}")
print(f"Files processed: {processed_files}")
print(f"Files skipped: {skipped_files}")

avg_rtf = total_proc / total_dur
avg_util = 100 / avg_rtf if avg_rtf > 0 else 0
print (f"avg_rtf: {avg_rtf}")
print (f"avg_util: {avg_util}")


print("\n=== Codebook Utilization Report ===")
avg_utilizations = []
avg_entropies = []

for p in range(num_pairs):
    used = len(pair_indices[p])
    total = grid_lens[p]
    util = 100 * used / total
    level = levels[2*p]

    # build histogram for entropy
    counts = np.bincount(list(pair_indices[p]), minlength=total)
    hist = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)

    H = entropy(hist, base=2)               # Shannon entropy
    H_norm = (H / np.log2(total)) * 100     # normalized to [0,100]%

    print(f"Pair {p} ({grid_type}, L={level}): "
          f"unique {used}/{total} → {util:.2f}%, "
          f"entropy utilization {H_norm:.2f}%")

    avg_utilizations.append(util)
    avg_entropies.append(H_norm)

# averages
print(f"\nAverage per-pair unique utilization: {np.mean(avg_utilizations):.2f}%")
print(f"Average per-pair entropy utilization: {np.mean(avg_entropies):.2f}%")

# overall
used_composite = len(composite_indices)
total_codebook_size = np.prod(grid_lens)
util_overall = 100 * used_composite / total_codebook_size
print(f"\nOverall composite utilization: {used_composite}/{total_codebook_size} "
      f"→ {util_overall:.5f}%")
