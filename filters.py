import cv2
import numpy as np
from enum import Enum


class FiltersOption(Enum):
    NO_FILTER = 0
    MEDIAN_FILTER = 1
    GAUSSIAN_FILTER = 3
    BILATERAL_FILTER = 4
    SOBEL_EDGE_DETECTOR = 5
    CANNY_EDGE_DETECTOR = 6
    LAPLACIAN_OF_GAUSSIAN = 9
    INTENSITY_NORMALIZATION = 10
    CLAHE = 11


class Filters:
    def __init__(self):
        self.lower = 70
        self.upper = 100

    def init_canny(self, frame, sigma):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Zero-parameter, automatic Canny edge detection with Python and OpenCV
        v = np.median(gray_image)

        # ---- Apply automatic Canny edge detection using the computed median----
        self.lower = int(max(0, (1.0 - sigma) * v))
        self.upper = int(min(255, (1.0 + sigma) * v))


    @staticmethod
    def median_filter(input_frame, kernel_size=3):
        """
        Applies median filtering.

        Args:
            input_frame: The input frame as a NumPy array.
            kernel_size: The size of the kernel for median filtering (default: 3).

        Returns:
            The filtered frame as a NumPy array.
        """

        filtered_frame = cv2.medianBlur(input_frame, kernel_size)

        return filtered_frame

    @staticmethod
    def gaussian_filter(input_frame, sigma=1.0, kernel_size=3):
        """
        Applies Gaussian filtering.

        Args:
            input_frame: The input frame as a NumPy array.
            sigma: The standard deviation of the Gaussian kernel (default: 1.0).
            kernel_size: The size of the kernel for Gaussian filtering (default: 3).

        Returns:
            The filtered frame as a NumPy array.
        """

        filtered_frame = cv2.GaussianBlur(input_frame, (kernel_size, kernel_size), sigma)

        return filtered_frame

    @staticmethod
    def average_filter(input_frame, kernel_size=3):
        """
        Applies average filtering.

        Args:
            input_frame: The input frame as a NumPy array.
            kernel_size: The size of the kernel for average filtering (default: 3).

        Returns:
            The filtered frame as a NumPy array.
        """

        filtered_frame = cv2.blur(input_frame, (kernel_size, kernel_size))

        return filtered_frame

    @staticmethod
    def bilateral_filter(input_frame, diameter=9, sigma_color=75, sigma_space=75):
        """
        Applies bilateral filtering.

        Args:
            input_frame: The input frame as a NumPy array.
            diameter: Diameter of each pixel neighborhood (default: 9).
            sigma_color: Filter sigma in the color space (default: 75).
            sigma_space: Filter sigma in the coordinate space (default: 75).

        Returns:
            The filtered frame as a NumPy array.
        """

        filtered_frame = cv2.bilateralFilter(input_frame, diameter, sigma_color, sigma_space)

        return filtered_frame

    @staticmethod
    def sobel_filter(input_frame):
        """
        Applies Sobel edge detection.

        Args:
            input_frame: The input frame as a NumPy array.

        Returns:
            The normalized gradient magnitude as a NumPy array.
        """

        # Convert the input frame to grayscale if it's in color
        if len(input_frame.shape) == 3:
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

        # Apply Sobel edge detector
        sobel_x = cv2.Sobel(input_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(input_frame, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Normalize the result to the range [0, 255]
        normalized_result = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

        return normalized_result.astype(np.uint8)

    def canny_detector(self, input_frame):
        """
        Applies Canny edge detection.

        Args:
            input_frame: The input frame as a NumPy array.
            low_threshold: The lower threshold for edge detection (default: 10).
            high_threshold: The higher threshold for edge detection (default: 30).

        Returns:
            The edges detected by the Canny edge detector as a NumPy array.
        """
        gaussian_kernel_size = 5


        # Convert the input frame to grayscale if it's in color
        if len(input_frame.shape) == 3:
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

        #blurred_frame = cv2.GaussianBlur(input_frame, (gaussian_kernel_size, gaussian_kernel_size), 0)
        # Apply Canny edge detector
        edges = cv2.Canny(input_frame, self.lower, self.upper)

        return edges

    @staticmethod
    def kirsch_filter(input_frame):
        """
        Applies Kirsch edge detection.

        Args:
            input_frame: The input frame as a NumPy array.

        Returns:
            The normalized gradient magnitude using Kirsch filters as a NumPy array.
        """

        # Convert the input frame to grayscale if it's in color
        if len(input_frame.shape) == 3:
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

        masks = [
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
        ]

        # Apply Kirsch filters for eight directions
        kirsch_results = [cv2.filter2D(input_frame, cv2.CV_64F, mask) for mask in masks]

        # Calculate the magnitude of the gradient
        gradient_magnitude = np.max(kirsch_results, axis=0)

        # Normalize the result to the range [0, 255]
        normalized_result = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

        return normalized_result.astype(np.uint8)

    @staticmethod
    def prewitt_filter(input_frame):
        """
        Applies Prewitt edge detection.

        Args:
            input_frame: The input frame as a NumPy array.

        Returns:
            The normalized gradient magnitude using Prewitt filters as a NumPy array.
        """

        # Convert the input frame to grayscale if it's in color
        if len(input_frame.shape) == 3:
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

        # Define Prewitt masks for the x and y directions
        prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        # Apply Prewitt filters in the x and y directions
        prewitt_x_result = cv2.filter2D(input_frame, cv2.CV_64F, prewitt_x)
        prewitt_y_result = cv2.filter2D(input_frame, cv2.CV_64F, prewitt_y)

        # Calculate the magnitude of the gradient
        gradient_magnitude = np.sqrt(prewitt_x_result ** 2 + prewitt_y_result ** 2)

        # Normalize the result to the range [0, 255]
        normalized_result = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

        return normalized_result.astype(np.uint8)

    @staticmethod
    def laplacian_of_gaussian(input_frame, kernel_size=3, sigma=1.0):
        """
        Applies Laplacian of Gaussian (LoG).

        Args:
            input_frame: The input frame as a NumPy array.
            kernel_size: The size of the kernel for Gaussian filtering (default: 3).
            sigma: The standard deviation of the Gaussian kernel (default: 1.0).

        Returns:
            The normalized result of LoG as a NumPy array.
        """

        blurred_frame = cv2.GaussianBlur(input_frame, (kernel_size, kernel_size), sigma)
        laplacian_result = cv2.Laplacian(blurred_frame, cv2.CV_64F)
        normalized_result = cv2.normalize(laplacian_result, None, 0, 255, cv2.NORM_MINMAX)

        return normalized_result.astype(np.uint8)

    @staticmethod
    def intensity_normalization(input_frame, kernel_size=7):
        """
        Performs intensity normalization on the input frame.

        Args:
            input_frame: The input frame as a NumPy array.
            kernel_size: The size of the neighborhood for calculating the mean and standard deviation (default: 7).

        Returns:
            The normalized frame as a NumPy array.
        """

        # Convert the input frame to grayscale if needed
        if len(input_frame.shape) == 3:
            gray_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = input_frame

        # Apply intensity normalization formula
        gauss_avg_values = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)
        std_neighbours_deviation = cv2.GaussianBlur(gray_frame ** 2, (kernel_size, kernel_size), 0)
        std_neighbours_deviation = np.sqrt(std_neighbours_deviation - gauss_avg_values ** 2)
        normalized_frame = (gray_frame - gauss_avg_values) / (std_neighbours_deviation + 0.001)

        # Convert back to color if needed
        if len(input_frame.shape) == 3:
            normalized_frame = normalized_frame.astype(np.float32)
            normalized_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_GRAY2BGR)
            #normalized_frame = normalized_frame.astype(np.uint8)
            return normalized_frame.astype(np.uint8)
        else:
            return normalized_frame

    @staticmethod
    def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

        # Apply CLAHE to the grayscale image
        clahe_image = clahe.apply(gray)

        return clahe_image
