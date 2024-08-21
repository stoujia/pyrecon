import os
import re
import pathlib
import subprocess
import csv


if __name__ == "__main__":


    MARSFET_DATAPATH = "/scratch/gauzias/data/datasets/MarsFet/derivatives/preprocessing"  

    MARSFET_MESO_SVORT_INIT = "/scratch/cmercier/results/svort/"
    MARSFET_MESO_ROSI = "/scratch/cmercier/results/rosi/"
    
    MARSFET_DATABASE = "/scratch/cmercier/code/pyrecon/marsfet_latest_participants.csv"

    input_data = MARSFET_MESO_SVORT_INIT
    output_data = MARSFET_MESO_ROSI
    stacks_path = MARSFET_DATAPATH
    csv_file =  MARSFET_DATABASE
    sub_list = []
    ses_list = []
    with open(csv_file,newline='') as csvfile:
        readrow = csv.reader(csvfile,delimiter=',')
        for row in readrow:
            sub_list.append(row[0])
            ses_list.append(row[1])

    subjects = [sub for sub in os.listdir(stacks_path) if os.path.isdir(os.path.join(stacks_path,sub))]
    subjects.sort()

    for subject in subjects:
        dir_subject = os.path.join(stacks_path, subject)
        sessions = os.listdir(dir_subject)
        for session in sessions:
            if subject == "sub-0567" and session == "ses-0681":
            #in sub_list and session in ses_list :
            #
                input_slices = os.path.join(input_data,subject, session, 'res')
                dir_out = os.path.join(output_data, subject, session)
                if True : 
                #not os.path.exists(os.path.join(dir_out,'res.joblib.gz')):
                    print('input_slices:',input_slices)
                    print('dir_output:',dir_out)
                    cmd = (
                        "sbatch"
                        + " "
                        + "/scratch/cmercier/code/pyrecon/ROSI/utils/slurm/rosi.slurm"
                        + " "
                        + input_slices
                        + " "
                        + dir_out
                        )
                    os.system(cmd)

