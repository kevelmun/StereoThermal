from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image, to_tensor
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance,  ImageFilter
import torch
import cv2

def aplicar_filtro_grayscale(imagen_tensor):
    """
    Convierte una imagen a escala de grises y replica los canales para mantener la compatibilidad de 3 canales.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.

    Returns:
        torch.Tensor: Tensor de la imagen filtrada en escala de grises con 3 canales.
    """
    imagen_pil = to_pil_image(imagen_tensor.cpu())
    imagen_grayscale = imagen_pil.convert("L")
    imagen_grayscale_tensor = torch.tensor(np.array(imagen_grayscale)).unsqueeze(0).repeat(3, 1, 1).float() / 255.0
    print(f"Grayscale Tensor - Min: {imagen_grayscale_tensor.min().item()}, "
          f"Max: {imagen_grayscale_tensor.max().item()}, "
          f"Mean: {imagen_grayscale_tensor.mean().item()}")
    return imagen_grayscale_tensor.to(imagen_tensor.device)

def aplicar_filtro_hsv(imagen_tensor, factor_saturacion=1.5):
    """
    Aumenta la saturación de una imagen en el espacio de color HSV.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.
        factor_saturacion (float, opcional): Factor para aumentar la saturación. Por defecto es 1.5.

    Returns:
        torch.Tensor: Tensor de la imagen filtrada con saturación aumentada.
    """
    imagen_pil = to_pil_image(imagen_tensor.cpu())
    imagen_hsv = imagen_pil.convert("HSV")
    h, s, v = imagen_hsv.split()
    s = s.point(lambda i: min(int(i * factor_saturacion), 255))
    imagen_hsv_modificada = Image.merge("HSV", (h, s, v))
    imagen_rgb = imagen_hsv_modificada.convert("RGB")
    imagen_rgb_tensor = to_tensor(imagen_rgb).to(imagen_tensor.device)
    print(f"HSV Tensor - Min: {imagen_rgb_tensor.min().item()}, "
          f"Max: {imagen_rgb_tensor.max().item()}, "
          f"Mean: {imagen_rgb_tensor.mean().item()}")
    return imagen_rgb_tensor

def aplicar_filtro_contraste(imagen_tensor, factor_contraste=1.5):
    """
    Ajusta el contraste de una imagen.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.
        factor_contraste (float, opcional): Factor por el cual se ajusta el contraste. >1 aumenta el contraste, <1 lo disminuye.

    Returns:
        torch.Tensor: Tensor de la imagen con contraste ajustado.
    """
    imagen_pil = to_pil_image(imagen_tensor.cpu())
    enhancer = ImageEnhance.Contrast(imagen_pil)
    imagen_contraste = enhancer.enhance(factor_contraste)
    imagen_contraste_tensor = to_tensor(imagen_contraste).to(imagen_tensor.device)
    print(f"Contrast Tensor - Min: {imagen_contraste_tensor.min().item()}, "
          f"Max: {imagen_contraste_tensor.max().item()}, "
          f"Mean: {imagen_contraste_tensor.mean().item()}")
    return imagen_contraste_tensor

def aplicar_filtro_blur(imagen_tensor, radio=2):
    """
    Aplica un desenfoque gaussiano a una imagen.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.
        radio (float, opcional): Radio del desenfoque.

    Returns:
        torch.Tensor: Tensor de la imagen con desenfoque aplicado.
    """
    imagen_pil = to_pil_image(imagen_tensor.cpu())
    imagen_blur = imagen_pil.filter(ImageFilter.GaussianBlur(radius=radio))
    imagen_blur_tensor = to_tensor(imagen_blur).to(imagen_tensor.device)
    print(f"Blur Tensor - Min: {imagen_blur_tensor.min().item()}, "
          f"Max: {imagen_blur_tensor.max().item()}, "
          f"Mean: {imagen_blur_tensor.mean().item()}")
    return imagen_blur_tensor

def aplicar_filtro_canny(imagen_tensor, threshold1=100, threshold2=200):
    """
    Aplica la detección de bordes Canny a una imagen.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.
        threshold1 (int, opcional): Primer umbral para la detección de bordes.
        threshold2 (int, opcional): Segundo umbral para la detección de bordes.

    Returns:
        torch.Tensor: Tensor de la imagen con bordes detectados.
    """
    imagen_np = imagen_tensor.cpu().permute(1, 2, 0).numpy() * 255.0
    imagen_np = imagen_np.astype(np.uint8)
    imagen_gray = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(imagen_gray, threshold1, threshold2)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_tensor = torch.tensor(edges_rgb).permute(2, 0, 1).float() / 255.0
    return edges_tensor.to(imagen_tensor.device)

def aplicar_filtro(imagen_tensor, tipo_filtro):
    """
    Aplica un filtro especificado a una imagen utilizando funciones dedicadas.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.
        tipo_filtro (str, opcional): Tipo de filtro a aplicar. Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none". Por defecto es "none".

    Returns:
        torch.Tensor: Tensor de la imagen filtrada.
    """
    tipo_filtro = tipo_filtro.lower() if tipo_filtro else "none"
    if tipo_filtro == "none":
        print("No se aplicará ningún filtro.")
        return imagen_tensor
    elif tipo_filtro == "grayscale":
        return aplicar_filtro_grayscale(imagen_tensor)
    elif tipo_filtro == "hsv":
        return aplicar_filtro_hsv(imagen_tensor)
    elif tipo_filtro == "contrast":
        return aplicar_filtro_contraste(imagen_tensor)
    elif tipo_filtro == "blur":
        return aplicar_filtro_blur(imagen_tensor)
    elif tipo_filtro == "canny":
        return aplicar_filtro_canny(imagen_tensor)
    else:
        raise ValueError(f"Filtro tipo '{tipo_filtro}' no es válido. "
                         f"Selecciona entre 'grayscale', 'hsv', 'blur', 'contrast', 'canny', 'none'.")

def calcular_distancia_media(pts0, pts1):
    """
    Calcula la distancia euclidiana media entre dos conjuntos de puntos.

    Args:
        pts0 (np.ndarray): Conjunto de puntos en la primera imagen.
        pts1 (np.ndarray): Conjunto de puntos en la segunda imagen.

    Returns:
        float: Distancia euclidiana media.
    """
    if len(pts0) == 0 or len(pts1) == 0:
        return float('inf')
    distancias = np.linalg.norm(pts0 - pts1, axis=1)
    return np.mean(distancias)

def evaluar_homografia(M, pts0, pts1):
    """
    Evalúa la homografía calculando el error de reproyección.

    Args:
        M (np.ndarray): Matriz de homografía.
        pts0 (np.ndarray): Puntos en la primera imagen (fixed).
        pts1 (np.ndarray): Puntos correspondientes en la segunda imagen (wrapped).

    Returns:
        float: Error de reproyección medio.
    """
    if M is None:
        return float('inf')
    pts0_homog = np.hstack([pts0, np.ones((pts0.shape[0], 1))])
    pts0_transformed = M.dot(pts0_homog.T).T
    pts0_transformed /= pts0_transformed[:, [2]]
    pts0_transformed = pts0_transformed[:, :2]
    errores = np.linalg.norm(pts0_transformed - pts1, axis=1)
    return np.mean(errores)

