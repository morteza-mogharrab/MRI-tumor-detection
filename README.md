# MRI Skull Image Analysis

## What the project does
This Python code provides a comprehensive pipeline for analyzing MRI skull images and identifying potential tumor regions. It is specifically designed for medical imaging analysis, with a focus on detecting potential tumors in MRI skull images. The code begins by importing essential libraries for image processing, including **NumPy** for array operations, **OpenCV (cv2)** for computer vision tasks, and **Matplotlib** for visualization. The core function, `sym_demo()`, orchestrates the processing steps, starting with prompting the user to input an MRI image file. Subsequently, it performs various image processing operations such as skull detection, segmentation, and calculation of Bhattacharyya coefficients to assess region similarity. The detected regions, including potential tumor locations, are then visualized alongside the original MRI image. Additionally, the code incorporates helper functions like `identify_skull()` for preprocessing and aligning skull images and `score()` for computing Bhattacharyya coefficients.

## Why the project is useful
This project serves a crucial role in medical imaging analysis, particularly in neurology and oncology. By automating the analysis process, it aids healthcare professionals in the early detection of tumors within the skull region. This capability can lead to timely interventions and improved patient outcomes.

## How users can get started with the project
To get started with the MRI Skull Image Analysis project, users should ensure they have Python installed on their system and the required libraries. The necessary library versions are as follows:
- **NumPy** version: 1.23.5
- **OpenCV** version: 4.8.1
- **sci-kit-image** version: 0.22.0
- **tkinter** version: 8.6
- **matplotlib** version: 3.8.2

Users can clone or download the project repository and run the main script. Upon execution, the program will prompt the user to select an MRI image file for analysis. Detailed instructions for running the program are provided within the code comments.

## Where users can get help with the project
Users can seek help with the MRI Skull Image Analysis project by referring to the comments within the code, which explain the various functions and processing steps. Additionally, users can find assistance and engage with the community by posting questions or issues on the project's GitHub repository or relevant forums and discussion boards related to medical imaging or Python programming.

## Who maintains and contributes to the project
The original code (Matlab Code) was developed by several researchers, namely Parth Sharma and Rakesh Sharma, as part of the "HMM Model for Brain Tumor Detection and Classification" project. The Python adaptation and further development of the code, including the version available here, has been undertaken by me. I was responsible for converting the original Matlab code to Python for this project.
