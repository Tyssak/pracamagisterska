import os
from moviepy.editor import VideoFileClip
import shutil
from pre_processor_one_face import PreProcessor
from filters import FiltersOption

def create_new_folder(path, sub_folder):
    new_path = os.path.join(path, sub_folder)

    try:
        os.makedirs(new_path)
        return new_path
    except OSError as e:
        print(f"Error creating folder: {e}")
        return None

# Define the directory containing the MKV files
input_directory = r"C:\RAVDESS dataset no neutral\test"

save_folder = r"C:\CLAHE\test"
filter_option = FiltersOption.CLAHE

# input_directory = r"C:\Users\kubac\OneDrive\Pictures\Z aparatu\test"
# filter_option = FiltersOption.NO_FILTER
# save_folder = r"C:\Users\kubac\OneDrive\Pictures\Z aparatu\test wynik2\three_frames"

substraction = False
dataset = 1 # 0 for devemo, 1 for RAVDESS, 2 for other

# Create output directory if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# Function to extract emotion from file name
def extract_emotion_from_devemo(file_name):
    # Split file name by underscores
    parts = file_name.split('_')
    # Emotion is the fourth element in the split parts
    return parts[4]

def extract_emotion_from_RAVDESS(file_name):
    # Split file name by underscores
    parts = file_name.split('-')
    # Emotion is the fourth element in the split parts
    return parts[2]

def is_emotion_strong_in_RAVDESS(file_name):
    # Split file name by underscores
    parts = file_name.split('-')
    # Emotion is the fourth element in the split parts
    if parts[3] == "01":
        return False
    else:
        return True

# Function to meintain the same actors in the files with similar name (to easier divide test and treinging group)
def rearrange_name_for_RAVDESS(file_name):
    # Split the file name by '.'
    name_parts = file_name.split('.')
    # Split the main name part by '-'
    main_name_parts = name_parts[0].split('-')
    # Get the last part
    last_part = main_name_parts[-1]
    # Remove the last part from the list
    main_name_parts.pop()
    # Insert the last part at the beginning
    main_name_parts.insert(0, last_part)
    # Join the main name parts back together with '-'
    new_main_name = '-'.join(main_name_parts)
    # Join the main name and extension back together
    new_name = new_main_name + '.' + name_parts[1]
    return new_name


def extract_frames(movie, times, imgdir, name):
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)

    clip = VideoFileClip(movie)
    for t in times:
        imgpath = os.path.join(imgdir, '{}_{}.png'.format(name, int(t * clip.fps)))
        clip.save_frame(imgpath, t)

subfolders = [f.name for f in os.scandir(input_directory) if f.is_dir()]

for subfolder in subfolders:
    input_folder = os.path.join(input_directory, subfolder)
    save_path = os.path.join(save_folder, subfolder)
    os.makedirs(save_path, exist_ok=True)
    for idx, file_name in enumerate(os.listdir(input_folder)):
        if (file_name.endswith(".mkv") or file_name.endswith(".mp4")): # and is_emotion_strong_in_RAVDESS(file_name):
            # Extract emotion from file name
            if dataset == 0:
                emotion = extract_emotion_from_devemo(file_name)
            elif  dataset == 1:
                emotion = extract_emotion_from_RAVDESS(file_name)
            else:
                emotion = subfolder
            # Get folder name based on emotion
            folder_name = emotion
            # if dataset == 1:
            #     emotion_folder = os.path.join(save_path, folder_name)
            # else:
            #     emotion_folder = save_path
            os.makedirs(save_path, exist_ok=True)
            # Read video file
            video_path = os.path.join(input_folder, file_name)
            #fd = FaceDetector()
            fd = PreProcessor()
            print(video_path)
            if dataset == 1:
                new_file_name = rearrange_name_for_RAVDESS(file_name)
            else:
                new_file_name = file_name
            fd.run_face_detection("Nagranie wideo", filter_option, video_path,
                                  frame_substraction=substraction, file_name = new_file_name, save_path = save_path)

print("Conversion complete.")
