import os
import glob
import shutil

def merging_folders(path, input_folders, output_folder):


    output_folder = os.path.join(path, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    slice_counter = 0

    for folder in input_folders:
        folder = os.path.join(path, folder)
        all_files = glob.glob(os.path.join(folder, "*.nii.gz"))

        # Separate and sort images and masks
        image_files = sorted([f for f in all_files if not (os.path.basename(f).startswith("mask_") and os.path.basename(f).endswith(".nii.gz"))],
                             key=lambda x: int(os.path.basename(x).split(".")[0]))
        mask_files = sorted([f for f in all_files if os.path.basename(f).startswith("mask_")],
                            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))

        for i, img_path in enumerate(image_files):
            filename = f"{slice_counter}.nii.gz"
            shutil.copy(img_path, os.path.join(output_folder, filename))
            mask_filename = f"mask_{slice_counter}.nii.gz"
            shutil.copy(mask_files[i], os.path.join(output_folder, mask_filename))
            slice_counter += 1

        # for i, mask_path in enumerate(mask_files):
        #     filename = f"mask_{slice_counter - len(mask_files) + i}.nii.gz"
        #     shutil.copy(mask_path, os.path.join(output_folder, filename))

    print(f"✅ All slices and masks have been merged into '{output_folder}'")


def append_foldername_to_renamed_files(root_dir):
    valid_exts = [".nii.gz", ".npy"]

    for sub_name in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub_name)

        if not os.path.isdir(sub_path):
            continue

        for scan_name in os.listdir(sub_path):  # e.g., petit1, moyen4, etc.
            scan_path = os.path.join(sub_path, scan_name)

            if not os.path.isdir(scan_path):
                continue

            for contrast in ["T1w", "T2w"]:
                contrast_path = os.path.join(scan_path, contrast)

                if not os.path.isdir(contrast_path):
                    continue

                for filename in os.listdir(contrast_path):
                    if filename.startswith('.'):
                        continue
                    if not (f"_{contrast}_" in filename):
                        continue
                    if f"_{scan_name}" in filename:
                        continue  # Already renamed
                    if not any(filename.endswith(ext) for ext in valid_exts):
                        continue

                    old_path = os.path.join(contrast_path, filename)

                    # Rename by inserting _{scan_name} before extension
                    if filename.endswith(".nii.gz"):
                        new_filename = filename.replace(".nii.gz", f"_{scan_name}.nii.gz")
                    elif filename.endswith(".npy"):
                        new_filename = filename.replace(".npy", f"_{scan_name}.npy")
                    else:
                        continue  # Shouldn't reach here

                    new_path = os.path.join(contrast_path, new_filename)

                    print(f"Renaming:\n  {old_path}\n  → {new_path}")
                    os.rename(old_path, new_path)

# # 🔁 Replace this with your actual root path
# root_directory = "/home/INT/jia.s/Bureau/sandbox/multi_simulated/"
# append_foldername_to_renamed_files(root_directory)

def fix_nomvt_placement(root_dir):
    valid_exts = [".nii.gz", ".npy"]

    for sub_name in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub_name)
        if not os.path.isdir(sub_path):
            continue

        for scan_name in os.listdir(sub_path):  # e.g., petit1, moyen4, etc.
            scan_path = os.path.join(sub_path, scan_name)
            if not os.path.isdir(scan_path):
                continue

            for contrast in ["T1w", "T2w"]:
                contrast_path = os.path.join(scan_path, contrast)
                if not os.path.isdir(contrast_path):
                    continue

                for filename in os.listdir(contrast_path):
                    if filename.startswith('.'):
                        continue
                    if "_nomvt_" not in filename:
                        continue  # Only fix files with misplaced _nomvt
                    if not any(filename.endswith(ext) for ext in valid_exts):
                        continue

                    old_path = os.path.join(contrast_path, filename)

                    # Strip extension
                    if filename.endswith(".nii.gz"):
                        base = filename[:-7]
                        ext = ".nii.gz"
                    elif filename.endswith(".npy"):
                        base = filename[:-4]
                        ext = ".npy"
                    else:
                        continue

                    # Remove _nomvt_, then add _nomvt back at the end
                    new_base = base.replace("_nomvt_", "_") + "_nomvt"
                    new_filename = new_base + ext
                    new_path = os.path.join(contrast_path, new_filename)

                    print(f"Renaming:\n  {old_path}\n  → {new_path}")
                    os.rename(old_path, new_path)