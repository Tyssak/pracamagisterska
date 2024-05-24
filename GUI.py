import os
import random
import tkinter as tk
from tkinter import filedialog
#from face_detection import FaceDetector
from pre_processor_classify import PreProcessor
#from face_detection_show_processing import FaceDetector
#from face_detection_no_avg import FaceDetector
#from pre_processor_one_face import PreProcessor
from filters import FiltersOption


class EmotionDetectionApp:
    def __init__(self, root):
        self.folder_path = None
        self.file_list = None
        self.root = root
        self.root.title("Wykrywanie emocji")
        self.root.geometry("400x200")

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(2, weight=1)

        self.choice_var = tk.StringVar()
        self.choice_var.set("Kamera")

        self.processing_option = tk.StringVar()
        self.processing_option.set("Brak filtru")

        self.is_random = tk.BooleanVar()
        self.is_random.trace_add("write", self.toggle_browse_button)

        self.options = ["Kamera", "Nagranie wideo", "Zdjęcia"]
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(len(self.options) + 4, weight=1)
        self.image_processing_options = {
            "Brak filtru": FiltersOption.NO_FILTER,
            "Median Filter": FiltersOption.MEDIAN_FILTER,
            #"Average Filter": FiltersOption.AVERAGE_FILTER,
            "Gaussian Filter": FiltersOption.GAUSSIAN_FILTER,
            "Bilateral Filter": FiltersOption.BILATERAL_FILTER,
            "Sobel Edge Detector": FiltersOption.SOBEL_EDGE_DETECTOR,
            "Canny Edge Detector": FiltersOption.CANNY_EDGE_DETECTOR,
            #"Kirsch Compass Mask": FiltersOption.KIRSCH_FILTER,
            #"Prewitt Edge Detector": FiltersOption.PREWITT_FILTER,
            "LOG Edge Detector": FiltersOption.LAPLACIAN_OF_GAUSSIAN,
            "CLAHE": FiltersOption.CLAHE,
            #"Face Features": FiltersOption.FACE_FEATURES
        }

        self.setup_ui()

    def setup_ui(self):
        choice_label = tk.Label(self.root, text="Wybierz tryb:")
        choice_label.grid(row=0, column=1, pady=5)

        for i, option in enumerate(self.options):
            radio_button = tk.Radiobutton(self.root, text=option, variable=self.choice_var, value=option)
            radio_button.grid(row=i + 1, column=1, padx=15, sticky="w")

        checkbox = tk.Checkbutton(self.root, text="Wybierz losowo", variable=self.is_random)
        checkbox.grid(row=1, column=2, pady=5, padx=5, sticky="w")

        self.path_var = tk.StringVar()
        path_entry = tk.Entry(self.root, textvariable=self.path_var, width=40)
        self.browse_button = tk.Button(self.root, text="Przeglądaj", command=self.browse_file)
        self.browse_button_random = tk.Button(self.root, text="Przeglądaj", command=self.browse_random_file)

        path_entry.grid(row=len(self.options) + 1, column=1, padx=0, pady=5, columnspan=2)
        self.browse_button.grid(row=len(self.options) + 1, column=3, padx=20, pady=5)

        processing_label = tk.Label(self.root, text="Wybierz filtr:")
        processing_label.grid(row=len(self.options) + 2, column=1, pady=5)

        processing_menu = tk.OptionMenu(self.root, self.processing_option, *self.image_processing_options)
        processing_menu.grid(row=len(self.options) + 2, column=2, pady=5, sticky="w")

        submit_button = tk.Button(self.root, text="Zatwierdź", command=self.on_submit)
        submit_button.grid(row=len(self.options) + 3, column=1, columnspan=2, pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        self.path_var.set(file_path)

    def browse_random_file(self):
        self.folder_path = filedialog.askdirectory()
        self.file_list = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]

        if self.file_list:
            random_file = random.choice(self.file_list)
            file_path = os.path.join(self.folder_path, random_file)
            self.path_var.set(file_path)
        else:
            self.path_var.set("No files in the selected folder")

    def on_submit(self):
        selected_option = self.choice_var.get()
        selected_filter_option = self.processing_option.get()
        if self.is_random.get():
            random_file = random.choice(self.file_list)
            file_path = os.path.join(self.folder_path, random_file)
        else:
            file_path = self.path_var.get()

        filter_option_int = self.image_processing_options.get(selected_filter_option)
        # face_detector = FaceDetector()
        # face_detector.run_face_detection(selected_option, filter_option_int, file_path, visualisation=True)
        face_detector = PreProcessor()
        face_detector.run_face_detection(selected_option, filter_option_int, file_path, frame_substraction=False)
        #face_detector.run_face_detection(selected_option, filter_option_int, file_path, visualization=True, frame_substraction=False)

    def toggle_browse_button(self, *args):
        if self.is_random.get():
            self.browse_button.grid_forget()
            self.browse_button_random.grid(row=len(self.options) + 1, column=3, padx=20, pady=5)
        else:
            self.browse_button_random.grid_forget()
            self.browse_button.grid(row=len(self.options) + 1, column=3, padx=20, pady=5)


if __name__ == '__main__':
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
