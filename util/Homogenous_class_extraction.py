import csv
import os
import shutil
import pandas as pd
import ipdb
def extract_and_store(full_id_col, target_classes, csv_file_path, root_dirs, output_base_dir):
    # Read the CSV and extract IDs matching the target classes
    matched_ids = {}
    for classes,subclass in zip(csv_file_path,target_classes):
        path = "/home/workspace/3DShape2VecSet/util/metadata/" + classes + ".csv" 
        csv_reader = pd.read_csv(path)
        
        # print(classes)

        for row in csv_reader.itertuples(index=False, name=None):
            # Check if the name in the row matches any target class name
            # print(row)
            fullId, wnsynset, wnlemmas, up, front, name, tags = row

            if subclass in wnlemmas.split(','):
                # Extract the identifier after the dot in fullId
                # print("yes")
                full_id = fullId.split('.')[-1]
                
                matched_ids[full_id + '.npy'] = classes  # Store matched name with ID
        
    # For each extracted ID, search in root directories
    for full_id, class_name in matched_ids.items():
        # Define the class-based output directory
        class_dir = os.path.join(output_base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
        # Search for files in root directories
        found = False
        # print(class_name)
        root_path = os.path.join(root_dirs,class_name)
        # print(root_path)
        for root, _, files in os.walk(root_path):
            for file_name in files:
                # ipdb.set_trace()
                if full_id in file_name:
                        # Copy the found file to the target class directory
                        # print("Hello")
                        src_path = os.path.join(root, file_name)
                        dest_path = os.path.join(class_dir, file_name)
                        shutil.copy2(src_path, dest_path)
                        found = True
                        print(f"Copied {file_name} to {class_dir}")
                        break  # Stop after finding the file
                if found:
                    break

# Define parameters
# subclasses = ["sniper rifle", "dresser", "park bench", "CRT screen", "subwoofer", "cellular telephone"]  # Example subclasses
# classes = ["04090263","02933112","02828884","03211117", "03691459", "04401088"]
subclasses = ["coupe"]
classes =["02958343"]
root_dir =  "/home/workspace/3DShape2VecSet/Dataset/ShapeNetV2_point" # Directories to search
output_base_dir = "/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetV2_point" 

# Run the function
extract_and_store(full_id_col="fullId", target_classes=subclasses,
                  csv_file_path=classes, root_dirs=root_dir, output_base_dir=output_base_dir)
