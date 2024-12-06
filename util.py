from pathlib import Path
from PIL import Image
import cv2
import numpy as np


def show_image(image: np.ndarray, name: str, mode: int = cv2.IMREAD_ANYCOLOR):
    # Si 'image' es una ruta de archivo, lee la imagen, si no es un array ya cargado.
    if isinstance(image, str):
        img = cv2.imread(image, mode)
    else:
        img = image  # Si ya es un array, no lo leemos de nuevo.
    # img_1 = cv2.applyColorMap(img_1, cv2.COLORMAP_JET)
    if img is None:
        print("Error al cargar la imagen.")
        return
    # blue, green, red = cv2.split(img)

    # # Crear una imagen donde solo se mantiene un canal (por ejemplo, el rojo)
    # canal = cv2.merge([blue, np.zeros_like(green), np.zeros_like(red)])
    # Mostrar la imagen
    cv2.imshow(name, img)
    cv2.waitKey(0)  # Espera hasta que se presione una tecla
    cv2.destroyAllWindows()  # Cierra todas las ventanas

def extract_channels(image, channels2keep):
    if max(channels2keep) >= image.shape[2]:
        raise ValueError("One or more channel indices are out of the image's channel range")
    filter_image = image[:,:,channels2keep]
    return filter_image

def mask_channles(image, channels2keep):
    masked_image = np.zeros_like(image)
    masked_image[:,:,channels2keep] = image[:,:,channels2keep]
    return masked_image

def save_image(image, directory_path, extension='png'):
    """
    Saves the given image to the specified directory with a sequential filename.

    Parameters:
    - image (PIL.Image.Image or numpy.ndarray): The image to save.
    - directory_path (str or Path): The directory where the image will be saved.
    - extension (str): The file extension (default is 'png').

    Returns:
    - Path: The path where the image was saved.
    """
    directory = Path(directory_path)
    directory.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Initialize the counter
    counter = 1

    while True:
        # Construct the filename
        filename = f"{counter}.{extension}"
        file_path = directory / filename

        if not file_path.exists():
            try:
                if isinstance(image, Image.Image):  # PIL Image
                    image.save(file_path)
                elif isinstance(image, (np.ndarray,)):  # OpenCV Image (numpy array)
                    cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    raise ValueError("Unsupported image type. Provide a PIL.Image or numpy.ndarray.")
                
                print(f"Image saved as '{file_path}'.")
                return file_path
            except Exception as e:
                print(f"Failed to save image: {e}")
                raise
        else:
            counter += 1