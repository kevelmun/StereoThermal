from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image, to_tensor
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance,  ImageFilter
import torch
import cv2

def apply_filter_grayscale(imagen_tensor):
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

def apply_filter_hsv(imagen_tensor, factor_saturacion=1.5):
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

def apply_filter_contraste(imagen_tensor, factor_contraste=1.5):
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

def apply_filter_blur(imagen_tensor, radio=2):
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

def apply_filter_canny(imagen_tensor, threshold1=100, threshold2=200):
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

def apply_filter(imagen_tensor, tipo_filtro):
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
        return apply_filter_grayscale(imagen_tensor)
    elif tipo_filtro == "hsv":
        return apply_filter_hsv(imagen_tensor)
    elif tipo_filtro == "contrast":
        return apply_filter_contraste(imagen_tensor)
    elif tipo_filtro == "blur":
        return apply_filter_blur(imagen_tensor)
    elif tipo_filtro == "canny":
        return apply_filter_canny(imagen_tensor)
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

def evaluate_homography(M, pts0, pts1):
    """
    Evaluates the homography by calculating the reprojection error.

    Args:
        M (np.ndarray): Homography matrix.
        pts0 (np.ndarray): Points in the first image (fixed).
        pts1 (np.ndarray): Corresponding points in the second image (wrapped).

    Returns:
        float: Mean reprojection error.
    """
    if M is None:
        return float('inf')
    pts0_homog = np.hstack([pts0, np.ones((pts0.shape[0], 1))])
    pts0_transformed = M.dot(pts0_homog.T).T
    pts0_transformed /= pts0_transformed[:, [2]]
    pts0_transformed = pts0_transformed[:, :2]
    errors = np.linalg.norm(pts0_transformed - pts1, axis=1)
    return np.mean(errors)



def filter_and_analyze_matches(matches, scores, threshold=0.75):
    """
    Filters reliable matches based on a score threshold and calculates statistics.

    Parameters:
    - matches: List of match objects.
    - scores: List or tensor of scores corresponding to each match.
    - threshold: Score threshold to determine reliable matches (default is 0.75).

    Returns:
    - reliable_points: List of reliable matches above the threshold.
    - average_score: Average score of all matches.
    - util_percentage_points: Percentage of matches that are reliable.
    """
    # Filter reliable points with a score >= threshold
    reliable_points = [match for match, score in zip(matches, scores) if score.item() >= threshold]

    # Calculate the average score
    average_score = scores.mean().item()
    print(f"The average score is: {average_score}\n")

    print(f"The number of points above a score threshold of {threshold} is: {len(reliable_points)}\n")

    util_percentage_points = (len(reliable_points) * 100) / len(matches)
    print("Therefore, for this registration:")
    print(f"{util_percentage_points:.2f}% of the points are precise")

    return reliable_points, average_score, util_percentage_points


def apply_afin_transformation(feats0, feats1, matches01, imagen0, imagen1, threshold=50):
    """
    Aplica una transformación afín para registrar imagen0 con respecto a imagen1.

    Parámetros:
    - feats0: Características extraídas de imagen0.
    - feats1: Características extraídas de imagen1.
    - matches01: Diccionario con los emparejamientos entre feats0 y feats1.
    - imagen0: Tensor de PyTorch de la primera imagen (C, H, W).
    - imagen1: Tensor de PyTorch de la segunda imagen (C, H, W).
    - threshold: Número mínimo de correspondencias requeridas para proceder.

    Retorna:
    - imagen0_warped: Imagen resultante después de aplicar la transformación afín.
    """

    # Extraer los emparejamientos y las puntuaciones
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir los puntos clave a NumPy y asegurarse de que sean float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # Verificar si hay suficientes correspondencias
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # Calcular la matriz de transformación afín
    M, inliers = cv2.estimateAffine2D(
        pts0,
        pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )

    if M is None:
        raise ValueError("No se pudo calcular la transformación afín.")

    # Convertir tensores a arrays de NumPy y transponer para obtener (H, W, C)
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    # Asegurarse de que las imágenes sean de tipo uint8
    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    # Aplicar la transformación afín
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    error = evaluate_homography(M, pts0, pts1)
    return imagen0_warped, points0, scores, error

def apply_homography_transformation(feats0, feats1, matches01, imagen0, imagen1, threshold=50):
    """
    Aplica una transformación de homografía para registrar imagen0 con respecto a imagen1.

    Parámetros:
    - feats0: Características extraídas de imagen0.
    - feats1: Características extraídas de imagen1.
    - matches01: Diccionario con los emparejamientos entre feats0 y feats1.
    - imagen0: Tensor de PyTorch de la primera imagen (C, H, W).
    - imagen1: Tensor de PyTorch de la segunda imagen (C, H, W).
    - threshold: Número mínimo de correspondencias requeridas para proceder.

    Retorna:
    - imagen0_warped: Imagen resultante послé de aplicar a transformación de homografia.
    """

    matches, scores = matches01["matches"], matches01["scores"]
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]
    
    pts0 = points0.cpu().numpy()
    pts1 = points1.cpu().numpy()



    if(len(pts0) < threshold):
            print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
            return None
    # Calcular la matriz de homografía
    M, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0) # Este 5 se puede subir hasta 10
    
    if M is None:
        raise ValueError("No se pudo calcular la homografía.")
    
    # Convertir tensor de PyTorch a imagen PIL para la transformación
    imagen0_pil = to_pil_image(imagen0.cpu())
    imagen1_pil = to_pil_image(imagen1.cpu())
    
    # Aplicar la transformación de perspectiva
    imagen0_warped = cv2.warpPerspective(
        np.array(imagen0_pil), 
        M, 
        (imagen1.shape[2], imagen1.shape[1])
    )
    error = evaluate_homography(M, pts0, pts1)
    return imagen0_warped, points0, scores, error


def apply_similarity_transformation(feats0, feats1, matches01, imagen0, imagen1, threshold=50):
    """
    Aplica una transformación de similaridad para registrar imagen0 con respecto a imagen1.

    Parámetros:
    - feats0: Características extraídas de imagen0.
    - feats1: Características extraídas de imagen1.
    - matches01: Diccionario con los emparejamientos entre feats0 y feats1.
    - imagen0: Tensor de PyTorch de la primera imagen (C, H, W).
    - imagen1: Tensor de PyTorch de la segunda imagen (C, H, W).
    - threshold: Número mínimo de correspondencias requeridas para proceder.

    Retorna:
    - imagen0_warped: Imagen resultante después de aplicar la transformación de similaridad.
    """
    # Extraer los emparejamientos y las puntuaciones
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir los puntos clave a NumPy y asegurarse de que sean float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # Verificar si hay suficientes correspondencias
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # Calcular la matriz de transformación de similaridad
    # Usamos estimateAffinePartial2D con fullAffine=False para restringir a transformación de similaridad
    M, inliers = cv2.estimateAffinePartial2D(
        pts0,
        pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )

    if M is None:
        raise ValueError("No se pudo calcular la transformación de similaridad.")

    # Convertir tensores a arrays de NumPy y transponer para obtener (H, W, C)
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    # Asegurarse de que las imágenes sean de tipo uint8
    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    # Aplicar la transformación de similaridad
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    error = evaluate_homography(M, pts0, pts1)
    return imagen0_warped, points0, scores, error


def apply_rigid_transformation(feats0, feats1, matches01, imagen0, imagen1, threshold=50):
    """
    Aplica una transformación rígida para registrar imagen0 con respecto a imagen1.

    Parámetros:
    - feats0: Características extraídas de imagen0.
    - feats1: Características extraídas de imagen1.
    - matches01: Diccionario con los emparejamientos entre feats0 y feats1.
    - imagen0: Tensor de PyTorch de la primera imagen (C, H, W).
    - imagen1: Tensor de PyTorch de la segunda imagen (C, H, W).
    - threshold: Número mínimo de correspondencias requeridas para proceder.

    Retorna:
    - imagen0_warped: Imagen resultante después de aplicar la transformación rígida.
    """
    # Extraer los emparejamientos y las puntuaciones
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir los puntos clave a NumPy y asegurarse de que sean float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # Verificar si hay suficientes correspondencias
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # Estimar la transformación rígida (rotación y traslación)
    # Calcular los centroides de los conjuntos de puntos
    centroid0 = np.mean(pts0, axis=0)
    centroid1 = np.mean(pts1, axis=0)

    # Restar los centroides para obtener coordenadas centradas
    pts0_centered = pts0 - centroid0
    pts1_centered = pts1 - centroid1

    # Calcular la matriz de covarianza
    H = np.dot(pts0_centered.T, pts1_centered)

    # Calcular la SVD de la matriz de covarianza
    U, S, Vt = np.linalg.svd(H)

    # Calcular la matriz de rotación
    R = np.dot(Vt.T, U.T)

    # Asegurarse de que la matriz de rotación es válida (determinante = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Calcular la traslación
    t = centroid1.T - np.dot(R, centroid0.T)

    # Construir la matriz de transformación 2x3 para cv2.warpAffine
    M = np.hstack((R, t.reshape(2, 1)))

    # Convertir tensores a arrays de NumPy y transponer para obtener (H, W, C)
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    # Asegurarse de que las imágenes sean de tipo uint8
    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    # Aplicar la transformación rígida
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    error = evaluate_homography(M, pts0, pts1)
    return imagen0_warped, points0, scores, error


def apply_rigid_transformation_ransac(feats0, feats1, matches01, imagen0, imagen1, threshold=50, ransac_iterations=1000, ransac_threshold=5.0):
    # Extraer los emparejamientos y las puntuaciones
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir los puntos clave a NumPy y asegurarse de que sean float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # Verificar si hay suficientes correspondencias
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    max_inliers = 0
    best_M = None

    for _ in range(ransac_iterations):
        # Seleccionar aleatoriamente 2 pares de puntos
        idx = np.random.choice(len(pts0), 2, replace=False)
        sample_pts0 = pts0[idx]
        sample_pts1 = pts1[idx]

        # Estimar la transformación rígida con los 2 pares seleccionados
        M_candidate = estimate_rigid_transform(sample_pts0, sample_pts1)

        # Aplicar la transformación a todos los puntos
        pts0_transformed = (np.dot(M_candidate[:, :2], pts0.T) + M_candidate[:, 2:3]).T

        # Calcular el error
        errors = np.linalg.norm(pts0_transformed - pts1, axis=1)

        # Contar el número de inliers
        inliers = errors < ransac_threshold
        num_inliers = np.sum(inliers)

        # Actualizar la mejor estimación
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_M = M_candidate
            best_inliers = inliers

    if best_M is None:
        raise ValueError("No se pudo calcular la transformación rígida.")

    # Recalcular la transformación usando todos los inliers
    M = estimate_rigid_transform(pts0[best_inliers], pts1[best_inliers])

    # Convertir tensores a arrays de NumPy y transponer para obtener (H, W, C)
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    # Asegurarse de que las imágenes sean de tipo uint8
    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    # Aplicar la transformación rígida
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    error = evaluate_homography(M, pts0, pts1)
    return imagen0_warped, points0, scores, error

def estimate_rigid_transform(pts0, pts1):
    # Implementación similar a la anterior
    centroid0 = np.mean(pts0, axis=0)
    centroid1 = np.mean(pts1, axis=0)
    pts0_centered = pts0 - centroid0
    pts1_centered = pts1 - centroid1
    H = np.dot(pts0_centered.T, pts1_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid1.T - np.dot(R, centroid0.T)
    M = np.hstack((R, t.reshape(2, 1)))
    return M

def apply_translation_transformation(feats0, feats1, matches01, imagen0, imagen1, threshold=50):
    """
    Aplica una transformación de traslación para registrar imagen0 con respecto a imagen1.

    Parámetros:
    - feats0: Características extraídas de imagen0.
    - feats1: Características extraídas de imagen1.
    - matches01: Diccionario con los emparejamientos entre feats0 y feats1.
    - imagen0: Tensor de PyTorch de la primera imagen (C, H, W).
    - imagen1: Tensor de PyTorch de la segunda imagen (C, H, W).
    - threshold: Número mínimo de correspondencias requeridas para proceder.

    Retorna:
    - imagen0_warped: Imagen resultante después de aplicar la transformación de traslación.
    """
    # Extraer los emparejamientos
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir los puntos clave a NumPy y asegurarse de que sean float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # Verificar si hay suficientes correspondencias
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # Calcular el vector de traslación como la media de las diferencias entre los puntos emparejados
    translations = pts1 - pts0
    t_x = np.median(translations[:, 0])
    t_y = np.median(translations[:, 1])

    # Construir la matriz de transformación 2x3 para cv2.warpAffine
    M = np.array([[1, 0, t_x],
                  [0, 1, t_y]], dtype=np.float32)

    # Convertir tensores a arrays de NumPy y transponer para obtener (H, W, C)
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    # Asegurarse de que las imágenes sean de tipo uint8
    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    # Aplicar la transformación de traslación
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    error = evaluate_homography(M, pts0, pts1)
    return imagen0_warped, points0, scores, error



def apply_translation_transformation_ransac(feats0, feats1, matches01, imagen0, imagen1, threshold=1, ransacReprojThreshold=5.0):
    """
    Aplica una transformación de traslación para registrar imagen0 con respecto a imagen1 utilizando RANSAC.

    Parámetros:
    - feats0: Características extraídas de imagen0.
    - feats1: Características extraídas de imagen1.
    - matches01: Diccionario con los emparejamientos entre feats0 y feats1.
    - imagen0: Tensor de PyTorch de la primera imagen (C, H, W).
    - imagen1: Tensor de PyTorch de la segunda imagen (C, H, W).
    - threshold: Número mínimo de correspondencias requeridas para proceder.
    - ransacReprojThreshold: Umbral de reproyección para RANSAC.

    Retorna:
    - imagen0_warped: Imagen resultante después de aplicar la transformación de traslación.
    """
    # Extraer los emparejamientos
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir los puntos clave a NumPy y asegurarse de que sean float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # Verificar si hay suficientes correspondencias
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # Añadir una dimensión extra para cv2.estimateAffine2D
    pts0_reshaped = pts0.reshape(-1, 1, 2)
    pts1_reshaped = pts1.reshape(-1, 1, 2)

    # Utilizar cv2.estimateAffine2D con RANSAC para estimar la traslación
    M, inliers = cv2.estimateAffine2D(
        pts0_reshaped,
        pts1_reshaped,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransacReprojThreshold,
        refineIters=10  # Puedes ajustar el número de iteraciones de refinamiento
    )

    if M is None:
        raise ValueError("No se pudo calcular la transformación de traslación.")

    # Extraer solo la traslación de la matriz M
    t_x = M[0, 2]
    t_y = M[1, 2]
    M_translation = np.array([[1, 0, t_x],
                              [0, 1, t_y]], dtype=np.float32)

    # Convertir tensores a arrays de NumPy y transponer para obtener (H, W, C)
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    # Asegurarse de que las imágenes sean de tipo uint8
    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    # Aplicar la transformación de traslación
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M_translation,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    error = evaluate_homography(M, pts0, pts1)
    return imagen0_warped, points0, scores, error


import torch.nn.functional as F

def apply_translation_torch(imagen, t_x, t_y):
    """
    Aplica una traslación pura a una imagen tensorial usando PyTorch.
    """
    _, _, h, w = imagen.shape

    # Crear una transformación afín para solo traslación
    M = torch.tensor(
        [[1, 0, t_x / (w / 2)],  # Normalización de las traslaciones a [-1, 1]
         [0, 1, t_y / (h / 2)]], 
        dtype=torch.float32, 
        device=imagen.device
    ).unsqueeze(0)

    # Crear la cuadrícula para la transformación
    grid = F.affine_grid(M, imagen.unsqueeze(0).size(), align_corners=False)

    # Aplicar la transformación
    return F.grid_sample(imagen.unsqueeze(0), grid, align_corners=False).squeeze(0)

def apply_translation_transformation2(feats0, feats1, matches01, imagen0, imagen1, threshold=50, umbral_puntuacion=0.5):
    """
    Aplica una transformación de traslación para registrar imagen0 con respecto a imagen1 usando PyTorch.

    Parámetros:
    - feats0: Características extraídas de imagen0.
    - feats1: Características extraídas de imagen1.
    - matches01: Diccionario con los emparejamientos entre feats0 y feats1.
    - imagen0: Tensor de PyTorch de la primera imagen (C, H, W).
    - imagen1: Tensor de PyTorch de la segunda imagen (C, H, W).
    - threshold: Número mínimo de correspondencias requeridas para proceder.
    - umbral_puntuacion: Puntuación mínima para considerar emparejamientos válidos.

    Retorna:
    - imagen0_warped: Imagen resultante después de aplicar la transformación de traslación.
    """
    # Extraer los emparejamientos
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Filtrar emparejamientos por puntuación
    valid_matches = scores > umbral_puntuacion
    points0 = points0[valid_matches]
    points1 = points1[valid_matches]

    if len(points0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(points0)} matches válidos.")
        return None

    # Convertir a NumPy
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # Calcular el vector de traslación
    translations = pts1 - pts0
    t_x = np.median(translations[:, 0])
    t_y = np.median(translations[:, 1])

    # Aplicar la traslación con PyTorch
    imagen0_warped = apply_translation_torch(imagen0, t_x, t_y)

    return imagen0_warped, points0, valid_matches.sum()