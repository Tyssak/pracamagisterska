import os
from pre_processor_prepare_dataset import PreProcessor
from filters import FiltersOption

def create_new_folder(path, sub_folder):
    """
    Creates a new folder at the specified path and sub-folder name.

    Args:
        path: The parent directory where the new folder will be created.
        sub_folder: The name of the sub-folder to create.

    Returns:
        new_path: The path of the newly created folder if successful, else None.
    """
    new_path = os.path.join(path, sub_folder)

    try:
        os.makedirs(new_path)
        return new_path
    except OSError as e:
        print(f"Error creating folder: {e}")
        return None

# Define the directory containing the MKV files
input_directory = r"G:\do_magisterki\datasety\FER2013\archive\test"
filter_option = FiltersOption.ADD_FRAMES

save_directory = r"G:\do_magisterki\datasety\FER2013\227\testy"

processed_images = 0
total_images = sum(len(files) for _, _, files in os.walk(input_directory))

# Get a list of subfolders (one for each emotion)
subfolders = [f.name for f in os.scandir(input_directory) if f.is_dir()]

for subfolder in subfolders:
    emotion_folder = os.path.join(input_directory, subfolder)
    save_folder = os.path.join(save_directory, subfolder)
    os.makedirs(save_folder, exist_ok=True)

    for idx, file_name in enumerate(os.listdir(emotion_folder)):
        img_path = os.path.join(emotion_folder, file_name)
        save_path = os.path.join(save_folder, file_name)
        fd = PreProcessor()
        #print(img_path)
        fd.run_face_detection("Zdjęcia", filter_option, img_path, file_name=file_name, save_path=save_path)
        processed_images += 1

    print(f"Processed {processed_images} of {total_images}")

print("Conversion complete.")
