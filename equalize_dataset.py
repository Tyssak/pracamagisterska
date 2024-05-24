import os
import random
import shutil


def select_random_files(source_folder, num_files):
    # Get a list of all files in the source folder
    all_files = os.listdir(source_folder)
    #print(len(all_files))
    # Check if the number of files in the folder is less than the required number
    if len(all_files) < num_files:
        print("Error: There are fewer files in the folder than the requested number.")
        return

    # Shuffle the list of files
    random.shuffle(all_files)

    # Select the first 'num_files' files
    selected_files = all_files[:num_files]

    return selected_files


def main():
    source_folder = r"F:\substraction10\test"  # Specify the path to your folder
    save_folder = r"G:\do_magisterki\datasety\RAVDESS\mixed\equalized\substraction10\test"
    #num_files = 2624  # Specify the number of files you want to select
    num_files = 704
    #num_files = 832
    #num_files = 3264
    #num_files = 5824
    #num_files = 1536

    subfolders = [f.name for f in os.scandir(source_folder) if f.is_dir()]

    for subfolder in subfolders:
        input_folder = os.path.join(source_folder, subfolder)
        save_path = os.path.join(save_folder, subfolder)
        os.makedirs(save_path, exist_ok=True)
        selected_files = select_random_files(input_folder, num_files)
        # Copy or process the selected files as needed
        for file_name in selected_files:
            file_path = os.path.join(input_folder, file_name)
            # For example, you can copy the selected files to another folder
            shutil.move(file_path, save_path)
            # Or perform any other operation you need

        print("Random files selected and copied successfully.")


if __name__ == "__main__":
    main()
