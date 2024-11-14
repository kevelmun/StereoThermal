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