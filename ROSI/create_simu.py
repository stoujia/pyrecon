import subprocess
import os

hr1 = "/envau/work/meca/data/dHCP/rel3_dhcp_anat_pipeline/sub-CC00053XX04/ses-8607/anat/sub-CC00053XX04_ses-8607_T1w.nii.gz"
hr2 = "/envau/work/meca/data/dHCP/rel3_dhcp_anat_pipeline/sub-CC00053XX04/ses-8607/anat/sub-CC00053XX04_ses-8607_T2w.nii.gz"
types = ["T1w", "T2w"]
mask = "/envau/work/meca/data/dHCP/rel3_dhcp_anat_pipeline/sub-CC00053XX04/ses-8607/anat/sub-CC00053XX04_ses-8607_desc-brain_mask.nii.gz"
base_output = "/home/INT/jia.s/Bureau/sandbox/multi_simulated/sub-CC00053XX04/ses-8607/translation_0"
simu_values = [1, 3, 5, 8, 10, 20, 30, 40, 50, 60, 70]
n_repeats = 2

for mv in simu_values:
    for rep in range(1, n_repeats + 1):
        sim_name = f"multi_motion{mv}_rep{rep}"
        output_dir = os.path.join(base_output, f"motion_{mv}", f"rep_{rep}")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python", "-m", "rosi.simulation.scriptSimulData_multi",
            "--hr1", hr1,
            "--hr2", hr2,
            "--types", *types,
            "--mask", mask,
            "--output", output_dir,
            "--rotation", str(mv), str(mv),
            "--translation", str(0), str(0),
            "--name", sim_name
        ]

        print(f"Launching simulation with rotation={mv}, rep={rep}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Simulation failed for rotation={mv}, rep={rep} with error:\n{e}")
