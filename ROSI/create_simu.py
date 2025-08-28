import os
import yaml
import subprocess

# Load YAML config
with open("steven/config_simu_supp.yaml", "r") as f:
    config = yaml.safe_load(f)

global_cfg = config["global"]
subjects = config["subjects"]

types = global_cfg["types"]
categories = global_cfg["categories"]
n_repeats = global_cfg["n_repeats"]
base_output = global_cfg["base_output"]

for subj_id, subj_info in subjects.items():
    mask = subj_info["mask"]
    hr_contrasts = subj_info["hr_contrasts"]

    for category, motion_val in categories.items():
        for rep in range(1, n_repeats + 1):
            for t in types:
                matched_contrast = None

                # Try to find the contrast path that includes the type (e.g., T1w or T2w)
                for contrast_name, path in hr_contrasts.items():
                    if t in path:
                        matched_contrast = path
                        break

                if matched_contrast is None:
                    print(f"⚠️  No matching HR contrast for type '{t}' in subject {subj_id}. Skipping...")
                    continue

                output_dir = os.path.join(base_output, subj_id, f"{category}{rep}", t)
                os.makedirs(output_dir, exist_ok=True)

                sim_name = f"{t}_{category}{rep}"

                cmd = [
                    "python", "-m", "rosi.simulation.scriptSimulData",
                    "--hr", matched_contrast,
                    "--mask", mask,
                    "--output", output_dir,
                    "--name", sim_name,
                    "--motion", str(-motion_val), str(motion_val)
                ]

                print(f"🚀 Launching simulation: {sim_name}")
                try:
                    print(cmd)
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Simulation failed for {sim_name}:\n{e}")
