import os
import random
from math import ceil

def split_and_save_file_list(root_folder, class_folders):
    for class_folder in class_folders:
        # Path to the class subfolder
        class_path = os.path.join(root_folder, class_folder)

        # Get all file names in the class folder
        file_names = [f for f in os.listdir(class_path) if f.endswith('.npz') and os.path.isfile(os.path.join(class_path, f))]   
        total_files = len(file_names)

        # Shuffle file names to ensure random splitting
        random.shuffle(file_names)

        # Calculate number of files for each split
        num_train = ceil(0.925 * total_files)
        remaining_files = total_files - num_train
        num_test = ceil((2 / 3) * remaining_files)
        num_val = remaining_files - num_test

        # Split file names into train, test, and val
        train_files = file_names[:num_train]
        test_files = file_names[num_train:num_train + num_test]
        val_files = file_names[num_train + num_test:]

        # Save each list to a file
        with open(os.path.join(class_path, 'train.lst'), 'w') as f_train, \
             open(os.path.join(class_path, 'test.lst'), 'w') as f_test, \
             open(os.path.join(class_path, 'val.lst'), 'w') as f_val:

            for file in train_files:
                f_train.write(file + '\n')
            for file in test_files:
                f_test.write(file + '\n')
            for file in val_files:
                f_val.write(file + '\n')

        # Print stats for confirmation
        print(f"Class: {class_folder}")
        print(f"Total files: {total_files}")
        print(f"Train files: {len(train_files)}")
        print(f"Test files: {len(test_files)}")
        print(f"Val files: {len(val_files)}\n")

# Example usage
root_folder = "/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetV2_point"
# class_folders = ["04090263","02933112","02828884","03211117", "03691459", "04401088"]  # replace with your class folder names
class_folders = ["02958343"]
split_and_save_file_list(root_folder, class_folders)
