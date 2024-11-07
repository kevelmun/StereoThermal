# Stereo and Thermal Vision Project

This repository contains the code and configuration for a vision system that integrates stereo and thermal cameras, with synchronized capture and tools for calibration and image registration. The system is designed for advanced image processing applications, providing simultaneous video and image capture from three cameras.

## Project Structure

- **Cameras/**
  - **cap_vt.py**: Script to capture synchronized video and images from the stereo and thermal cameras.
  - **test_camera.py**: Script for testing the connected camera IDs and validating the available resolutions.
  - **captures/**: Directory where captured images and videos are stored.

- **Image_registration/**
  - **lightglue/**: Contains the LightGlue tool for image registration.
  - **resultados/**: Folder where the image registration results are saved.
  - **registration.py**: Script for registering images between the stereo and thermal cameras.
  - **script.py**: Additional script used in the registration process.
  - **util.py**: Utility functions for image processing and registration.

- **requirements.txt**: Lists all required dependencies to run the project.
- **README.md**: Project documentation.

## Project Features

1. **Synchronized Capture**:
   - The system allows for synchronous image and video capture from three cameras (two stereo cameras and one thermal camera).
   - Includes a temporal capture system that ensures visual and thermal data are aligned in the same timeframe, ideal for applications requiring synchronized data from both visual and thermal sources.

2. **Camera ID and Resolution Testing**:
   - The `test_camera.py` script verifies the ID of each connected camera and validates the resolutions available, ensuring compatibility with system requirements.

3. **Calibration and Image Registration** *(in progress)*:
   - Calibration for all three cameras is being developed to ensure data overlay accuracy.
   - Utilizes the `lightglue` tool in the `Image_registration` folder for image registration, aligning images captured by the stereo and thermal cameras.

## Installation

To install and set up the environment, ensure you have the following dependencies:

- **Python >= 3.8**
- Libraries:
  - `opencv-python`
  - `numpy`
  - `lightglue` (for image registration, available in the `Image_registration/lightglue` folder)

Install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

### Image and Video Capture

Run the following command to start synchronized capture:

```bash
python Cameras/cap_vt.py
```

This script will save images and videos in the `Cameras/captures` directory.

### Camera ID and Resolution Testing

To test each camera's ID and validate available resolutions, use:

```bash
python Cameras/test_camera.py
```

This script provides a report of connected cameras, their IDs, and compatible resolutions.

### Calibration and Image Registration

The calibration and image registration process is under active development. To test image registration with `lightglue`, use:

```bash
python Image_registration/registration.py
```

The results will be saved in the `Image_registration/resultados` folder.

## Contributions

Contributions are welcome. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`feature/new-feature`).
3. Make your changes and commit (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

## Project Status

- **Current**: Working on camera calibration and image registration using LightGlue.
- **Main**: Stable production-ready synchronized capture, camera ID testing, and resolution validation.
