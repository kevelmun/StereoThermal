import cv2
import numpy as np

def fusion_bgr_lwir(imagen_bgr, imagen_lwir):
    """
    Fusiona una imagen bgr y una imagen LWIR en una imagen de 4 canales.
    Nota: El canal LWIR en un valor entero
    Parámetros:
    - imagen_bgr: numpy array de dimensiones (alto, ancho, 3)
    - imagen_lwir: numpy array de dimensiones (alto, ancho)

    Retorna:
    - imagen_fusionada: numpy array de dimensiones (alto, ancho, 4)
    """
    # Verificar que la imagen bgr tenga 3 canales

    if isinstance(imagen_bgr, str):
        imagen_bgr = cv2.imread(imagen_bgr, cv2.IMREAD_COLOR)

    if isinstance(imagen_lwir, str):
        imagen_lwir = cv2.imread(imagen_lwir, cv2.IMREAD_GRAYSCALE)
    if len(imagen_lwir.shape) == 3:
        imagen_lwir = cv2.cvtColor(imagen_lwir, cv2.COLOR_BGR2GRAY)
        
    if imagen_bgr.shape[2] != 3:
        raise ValueError("La imagen bgr debe tener 3 canales.")

    # Verificar que las dimensiones espaciales de las imágenes coincidan
    if imagen_bgr.shape[0] != imagen_lwir.shape[0] or imagen_bgr.shape[1] != imagen_lwir.shape[1]:
        raise ValueError("Las dimensiones de las imágenes no coinciden.")

    # Expandir la dimensión de la imagen LWIR para que tenga un tercer eje
    imagen_lwir_exp = np.expand_dims(imagen_lwir, axis=2)

    # Fusionar las imágenes concatenando en el eje de canales
    imagen_fusionada = np.concatenate((imagen_bgr, imagen_lwir_exp), axis=2)

    return imagen_fusionada


def fuse_bgr_thermal(bgr_image, thermal_image):
    """
    Fuses an bgr image and a thermal (LWIR) image into a single bgrT image.
    Note: Thermal channel goes from 0 to 1
    Parameters:
    bgr_image (numpy.ndarray): The bgr image as a 3D numpy array (height, width, 3)
    thermal_image (numpy.ndarray): The thermal (LWIR) image as a 2D numpy array (height, width)
    
    Returns:
    numpy.ndarray: The fused bgrT image as a 3D numpy array (height, width, 4)
    """
    # Ensure both input images have the same dimensions
    if isinstance(bgr_image, str):
        bgr_image = cv2.imread(bgr_image, cv2.IMREAD_COLOR)

    if isinstance(thermal_image, str):
        thermal_image = cv2.imread(thermal_image, cv2.IMREAD_GRAYSCALE)
    if len(thermal_image.shape) == 3:
        thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)

    if bgr_image.shape[:2] != thermal_image.shape:
        raise ValueError("bgr and thermal images must have the same dimensions")
    
    # Normalize the thermal image to the range [0, 1]
    thermal_image_norm = (thermal_image - thermal_image.min()) / (thermal_image.max() - thermal_image.min())
    
    # Create the fused bgrT image by stacking the bgr and thermal channels
    fused_image = np.dstack((bgr_image, thermal_image_norm))
    
    return fused_image
