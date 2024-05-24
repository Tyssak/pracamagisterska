import os
import random
import customtkinter as tk
from customtkinter import filedialog
# from face_detection import FaceDetector
from pre_processor_classify import PreProcessor
#from face_detection_show_processing import PreProcessor
from filters import FiltersOption


class EmotionDetectionApp:
    def __init__(self, root):
        self.browse_button_random = None
        self.browse_button = None
        self.model_browse_button = None
        self.model_path_var = None
        self.source_browse_button_random = None
        self.source_browse_button = None
        self.source_path_var = None
        self.folder_path = None
        self.file_list = None
        self.root = root
        self.root.title("Wykrywanie emocji")
        self.root.geometry("500x300")

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
        self.root.rowconfigure(len(self.options) + 5, weight=1)
        self.image_processing_options = {
            "Brak filtru": FiltersOption.NO_FILTER,
            "Median Filter": FiltersOption.MEDIAN_FILTER,
            # "Average Filter": FiltersOption.AVERAGE_FILTER,
            "Gaussian Filter": FiltersOption.GAUSSIAN_FILTER,
            "Bilateral Filter": FiltersOption.BILATERAL_FILTER,
            "Sobel Edge Detector": FiltersOption.SOBEL_EDGE_DETECTOR,
            "Canny Edge Detector": FiltersOption.CANNY_EDGE_DETECTOR,
            # "Kirsch Compass Mask": FiltersOption.KIRSCH_FILTER,
            # "Prewitt Edge Detector": FiltersOption.PREWITT_FILTER,
            "LOG Edge Detector": FiltersOption.LAPLACIAN_OF_GAUSSIAN,
            "CLAHE": FiltersOption.CLAHE,
            # "Face Features": FiltersOption.FACE_FEATURES
        }

        self.setup_ui()

    def setup_ui(self):
        choice_label = tk.CTkLabel(self.root, text="Wybierz tryb:")
        choice_label.grid(row=1, column=1)

        for i, option in enumerate(self.options):
            radio_button = tk.CTkRadioButton(self.root, text=option, variable=self.choice_var, value=option)
            radio_button.grid(row=i + 1, column=2, padx=5, sticky="w")

        checkbox = tk.CTkCheckBox(self.root, text="Losowy wybór zdjęcia/\n/nagrania z folderu", variable=self.is_random)
        checkbox.grid(row=1, column=4, padx=5, sticky="w")

        self.source_path_var = tk.StringVar()
        source_entry_label = tk.CTkLabel(self.root, text="Wybierz zdjęcie/nagranie:")
        source_entry_label.grid(row=len(self.options) + 1, column=1, padx=15, pady=5)
        source_path_entry = tk.CTkEntry(self.root, textvariable=self.source_path_var, width=200)
        self.source_browse_button = tk.CTkButton(self.root, text="Przeglądaj", command=self.source_browse_file)
        self.source_browse_button_random = tk.CTkButton(self.root, text="Przeglądaj",
                                                        command=self.browse_random_file)

        source_path_entry.grid(row=len(self.options) + 1, column=2, padx=0, pady=5, columnspan=2)
        self.source_browse_button.grid(row=len(self.options) + 1, column=4, padx=20, pady=5)

        self.model_path_var = tk.StringVar()
        model_entry_label = tk.CTkLabel(self.root, text="Wybierz model:")
        model_entry_label.grid(row=len(self.options) + 2, column=1, padx=15, pady=5)
        model_path_entry = tk.CTkEntry(self.root, placeholder_text="Domyślny", textvariable=self.model_path_var, width=200)
        self.model_browse_button = tk.CTkButton(self.root, text="Przeglądaj", command=self.model_browse_file)

        model_path_entry.grid(row=len(self.options) + 2, column=2, padx=0, pady=5, columnspan=2)
        self.model_browse_button.grid(row=len(self.options) + 2, column=4, padx=20, pady=5)
        self.model_path_var.set("Domyślny")

        processing_label = tk.CTkLabel(self.root, text="Wybierz filtr:")
        processing_label.grid(row=len(self.options) + 3, column=1, pady=5)

        processing_menu = tk.CTkOptionMenu(master=self.root, values=list(self.image_processing_options.keys()),
                                           command=self.processing_option.set)
        processing_menu.grid(row=len(self.options) + 3, column=2, pady=5, sticky="w")

        submit_button = tk.CTkButton(self.root, text="Zatwierdź", command=self.on_submit)
        submit_button.grid(row=len(self.options) + 4, column=1, columnspan=2, pady=10)

    def source_browse_file(self):
        file_path = filedialog.askopenfilename()
        self.source_path_var.set(file_path)

    def model_browse_file(self):
        file_path = filedialog.askopenfilename()
        self.model_path_var.set(file_path)

    def browse_random_file(self):
        self.folder_path = filedialog.askdirectory()
        self.file_list = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]

        if self.file_list:
            random_file = random.choice(self.file_list)
            file_path = os.path.join(self.folder_path, random_file)
            self.source_path_var.set(file_path)
        else:
            self.source_path_var.set("No files in the selected folder")


    def on_submit(self):
        selected_option = self.choice_var.get()
        selected_filter_option = self.processing_option.get()
        if self.is_random.get():
            random_file = random.choice(self.file_list)
            file_path = os.path.join(self.folder_path, random_file)
        else:
            file_path = self.source_path_var.get()

        filter_option_int = self.image_processing_options.get(selected_filter_option)
        # face_detector = FaceDetector()
        # face_detector.run_face_detection(selected_option, filter_option_int, file_path, visualisation=True)
        face_detector = PreProcessor()
        # if self.model_path_var.get() != "Domyślny":
        #     face_detector.run_face_detection(selected_option, filter_option_int, file_path, model_path = self.model_path_var.get(), frame_substraction=False)
        # else:
        #     face_detector.run_face_detection(selected_option, filter_option_int, file_path, frame_substraction=False)
        face_detector.run_face_detection(selected_option, filter_option_int, file_path, frame_substraction=False)

        # face_detector.run_face_detection(selected_option, filter_option_int, file_path, visualization=True, frame_substraction=False)

    def toggle_browse_button(self, *args):
        if self.is_random.get():
            self.source_browse_button.grid_forget()
            self.source_browse_button_random.grid(row=len(self.options) + 1, column=4, padx=20, pady=5)
        else:
            self.source_browse_button_random.grid_forget()
            self.source_browse_button.grid(row=len(self.options) + 1, column=4, padx=20, pady=5)


if __name__ == '__main__':
    tk.set_appearance_mode("dark")
    tk.set_default_color_theme("dark-blue")
    root = tk.CTk()
    app = EmotionDetectionApp(root)
    root.mainloop()
