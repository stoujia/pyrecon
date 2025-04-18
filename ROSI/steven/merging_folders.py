import os
import shutil
import glob

# Input folders

def merging_folders(path, input_folders, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    # Counter for new slice index
    slice_counter = 0
    output_folder = os.path.join(path, output_folder)

    for folder in input_folders:
        folder= os.path.join(path, folder)
        image_files = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))
        
        # Separate images and masks
        image_files = [f for f in image_files if not os.path.basename(f).startswith("mask_")]
        mask_files = [f for f in image_files if "mask_" in os.path.basename(f)]
        
        for img_path in image_files:
            filename = f"{slice_counter}.nii.gz"
            shutil.copy(img_path, os.path.join(output_folder, filename))
            slice_counter += 1
        
        for mask_path in mask_files:
            filename = f"mask_{slice_counter - len(mask_files) + mask_files.index(mask_path)}.nii.gz"
            shutil.copy(mask_path, os.path.join(output_folder, filename))

    print(f"✅ All slices and masks have been merged into '{output_folder}'")
