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
