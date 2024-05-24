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
input_directory = r"D:\do_magisterki\dane\RAVDESS\RAVDESS dataset"

save_path = r"D:\do_magisterki\dane\RAVDESS\RAVDESS strong"
# Create output directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

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


for idx, file_name in enumerate(os.listdir(input_directory)):
    if (file_name.endswith(".mkv") or file_name.endswith(".mp4")) and not is_emotion_strong_in_RAVDESS(file_name):
        # Extract emotion from file name
        emotion = extract_emotion_from_RAVDESS(file_name)
        # Get folder name based on emotion
        folder_name = emotion
        # Create folder if it doesn't exist
        emotion_folder = os.path.join(save_path, folder_name)
        os.makedirs(emotion_folder, exist_ok=True)
        # Read video file
        video_path = os.path.join(input_directory, file_name)
        #fd = FaceDetector()
        fd = PreProcessor()
        print(video_path)
        new_file_name = rearrange_name_for_RAVDESS(file_name)
        new_video_path = os.path.join(emotion_folder, new_file_name)
        # SAVE THE VIDEO TO NEW LOCATION
        shutil.move(video_path, new_video_path)

print("Conversion complete.")
