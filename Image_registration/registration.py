import PIL
import torch
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet, viz2d
from lightglue.utils import load_image, rbd
from torchvision.transforms.functional import resize, to_pil_image, to_tensor
from torchvision.utils import save_image
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from util import *
import seaborn as sns
import glob

def procesar_imagenes(
    ruta_imagen0,
    ruta_imagen1,
    extractor_tipo="superpoint",
    max_num_keypoints=4096,
    dispositivo=None,
    filtro_imagen0=None,  # Nuevo parámetro para aplicar filtros a imagen0
    filtro_imagen1=None,  # Nuevo parámetro para aplicar filtros a imagen1
    threshold= 0
):
    """
    Procesa dos imágenes para obtener una imagen resultante tras aplicar la transformación de perspectiva.

    Args:
        ruta_imagen0 (str): Ruta al primer archivo de imagen.
        ruta_imagen1 (str): Ruta al segundo archivo de imagen.
        extractor_tipo (str, opcional): Tipo de extractor a utilizar. Opciones: "superpoint", "disk", "sift", "aliked", "doghardnet". Por defecto es "superpoint".
        max_num_keypoints (int, opcional): Número máximo de puntos clave a extraer. Por defecto es 4096.
        dispositivo (torch.device, opcional): Dispositivo para ejecutar el modelo. Si es None, se selecciona automáticamente GPU si está disponible.
        filtro_imagen0 (str, opcional): Tipo de filtro a aplicar a imagen0. Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none". Por defecto es None.
        filtro_imagen1 (str, opcional): Tipo de filtro a aplicar a imagen1. Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none". Por defecto es None.

    Returns:
        PIL.Image: Imagen resultante tras la transformación de perspectiva.
    """
    
    # Configurar el dispositivo
    if dispositivo is None:
        dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Dispositivo utilizado:", dispositivo)
    
    # Inicializar extractor y matcher según el tipo especificado
    extractor_dict = {
        "superpoint": (SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(dispositivo), LightGlue(features='superpoint').eval().to(dispositivo)),
        "disk": (DISK(max_num_keypoints=max_num_keypoints).eval().to(dispositivo), LightGlue(features='disk').eval().to(dispositivo)),
        "sift": (SIFT(max_num_keypoints=max_num_keypoints).eval().to(dispositivo), LightGlue(features='sift').eval().to(dispositivo)),
        "aliked": (ALIKED(max_num_keypoints=max_num_keypoints).eval().to(dispositivo), LightGlue(features='aliked').eval().to(dispositivo)),
        "doghardnet": (DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(dispositivo), LightGlue(features='doghardnet').eval().to(dispositivo)),
    }
    
    if extractor_tipo.lower() not in extractor_dict:
        raise ValueError(f"Extractor tipo '{extractor_tipo}' no es válido. Selecciona entre 'superpoint', 'disk', 'sift', 'aliked', 'doghardnet'.")
    
    extractor, matcher = extractor_dict[extractor_tipo.lower()]
    
    try:
        # Cargar y preparar las imágenes
        imagen0 = load_image(ruta_imagen0).to(dispositivo)
        imagen1 = load_image(ruta_imagen1).to(dispositivo)
        
        # Aplicar los filtros a cada imagen si se especifica
        imagen0 = aplicar_filtro(imagen0, filtro_imagen0)
        imagen1 = aplicar_filtro(imagen1, filtro_imagen1)
        
        # Extraer características
        feats0 = extractor.extract(imagen0)
        feats1 = extractor.extract(imagen1)
        
        # Emparejar características
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        

        # # Obtener los puntos clave y las correspondencias del primer conjunto
        # kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        # m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
        matches, scores = matches01["matches"], matches01["scores"]
        points0 = feats0['keypoints'][matches[..., 0]]
        points1 = feats1['keypoints'][matches[..., 1]]
        
        pts0 = points0.cpu().numpy()
        pts1 = points1.cpu().numpy()
        
        if(len(pts0) < threshold):
                print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
                return None
        # Calcular la matriz de homografía
        M, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        
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
        
        # imagen0_warped_pil = Image.fromarray(imagen0_warped)
        
        # # Convertir la imagen warpada a tensor
        # imagen0_warped_tensor = to_tensor(imagen0_warped_pil).to(dispositivo)
        
        # # Redimensionar la imagen warpada si es necesario
        # imagen0_warped = resize(imagen0_warped, (480, 640))
        
        # # Convertir a PIL para retornar
        # resultado_pil = to_pil_image(imagen0_warped_tensor.cpu())
        
        print(f"shape: {imagen0_warped.shape} - Matches: {len(points0)} - Score: {len(scores)}")
        return imagen0_warped

    except Exception as e:
        print(f"Error al procesar las imágenes: {e}")
        return None, None

def mostrar_correspondencias_mejorada(
    ruta_imagen0,
    ruta_imagen1,
    extractor_tipo="superpoint",
    max_num_keypoints=4096,
    dispositivo=None,
    umbral_score=0.5,
    guardar_figura=False,
    ruta_guardado="correspondencias_mejoradas.png",
    filtro_imagen0=None,        # Nuevo parámetro para aplicar filtros a imagen0
    filtro_imagen1=None,        # Nuevo parámetro para aplicar filtros a imagen1
    mostrar_correspondencias=True,  # Nuevo parámetro para activar/desactivar las líneas de correspondencia
    mostrar_metricas=True           # Nuevo parámetro para ocultar/mostrar las métricas
):
    """
    Muestra la correspondencia de los puntos clave entre dos imágenes después de aplicar una transformación de perspectiva.
    Además, distingue visualmente entre correspondencias inliers y outliers y reporta métricas de evaluación.

    Args:
        ruta_imagen0 (str): Ruta al primer archivo de imagen.
        ruta_imagen1 (str): Ruta al segundo archivo de imagen.
        extractor_tipo (str, opcional): Tipo de extractor a utilizar. Opciones: "superpoint", "disk", "sift", "aliked", "doghardnet". Por defecto es "superpoint".
        max_num_keypoints (int, opcional): Número máximo de puntos clave a extraer. Por defecto es 4096.
        dispositivo (torch.device, opcional): Dispositivo para ejecutar el modelo. Si es None, se selecciona automáticamente GPU si está disponible.
        umbral_score (float, opcional): Umbral de score para filtrar las correspondencias antes de visualizarlas. Valores entre 0 y 1. Por defecto es 0.5.
        guardar_figura (bool, opcional): Si se establece en True, guarda la figura en la ruta especificada. Por defecto es False.
        ruta_guardado (str, opcional): Ruta donde se guardará la figura si `guardar_figura` es True. Por defecto es "correspondencias_mejoradas.png".
        filtro_imagen0 (str, opcional): Tipo de filtro a aplicar a imagen0. Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none". Por defecto es None.
        filtro_imagen1 (str, opcional): Tipo de filtro a aplicar a imagen1. Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none". Por defecto es None.
        mostrar_correspondencias (bool, opcional): Si se establece en True, dibuja las líneas de correspondencia entre keypoints. Por defecto es True.
        mostrar_metricas (bool, opcional): Si se establece en True, muestra las métricas de evaluación en la figura y la consola. Por defecto es True.

    Returns:
        None. Muestra una figura con las correspondencias entre los puntos clave de las dos imágenes, diferenciando inliers y outliers y mostrando métricas de evaluación si está habilitado.
    """
    
    # Configurar el dispositivo
    if dispositivo is None:
        dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Dispositivo utilizado:", dispositivo)
    
    # Inicializar extractor y matcher según el tipo especificado
    extractor_dict = {
        "superpoint": (SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(dispositivo), LightGlue(features='superpoint').eval().to(dispositivo)),
        "disk": (DISK(max_num_keypoints=max_num_keypoints).eval().to(dispositivo), LightGlue(features='disk').eval().to(dispositivo)),
        "sift": (SIFT(max_num_keypoints=max_num_keypoints).eval().to(dispositivo), LightGlue(features='sift').eval().to(dispositivo)),
        "aliked": (ALIKED(max_num_keypoints=max_num_keypoints).eval().to(dispositivo), LightGlue(features='aliked').eval().to(dispositivo)),
        "doghardnet": (DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(dispositivo), LightGlue(features='doghardnet').eval().to(dispositivo)),
    }
    
    if extractor_tipo.lower() not in extractor_dict:
        raise ValueError(f"Extractor tipo '{extractor_tipo}' no es válido. Selecciona entre 'superpoint', 'disk', 'sift', 'aliked', 'doghardnet'.")
    
    extractor, matcher = extractor_dict[extractor_tipo.lower()]
    
    try:
        # Verificar que las rutas sean cadenas de texto
        if not isinstance(ruta_imagen0, (str, os.PathLike)) or not isinstance(ruta_imagen1, (str, os.PathLike)):
            raise TypeError("Las rutas de las imágenes deben ser cadenas de texto, bytes o objetos Path.")
        
        # Verificar que las rutas existan
        if not os.path.exists(ruta_imagen0):
            raise FileNotFoundError(f"La ruta de la imagen0 no existe: {ruta_imagen0}")
        if not os.path.exists(ruta_imagen1):
            raise FileNotFoundError(f"La ruta de la imagen1 no existe: {ruta_imagen1}")
        
        # Cargar y preparar las imágenes
        imagen0 = load_image(ruta_imagen0).to(dispositivo)
        imagen1 = load_image(ruta_imagen1).to(dispositivo)
        
        # Aplicar los filtros a cada imagen si se especifica
        imagen0 = aplicar_filtro(imagen0, filtro_imagen0)
        imagen1 = aplicar_filtro(imagen1, filtro_imagen1)
        
        # Extraer características del primer conjunto de imágenes
        feats0 = extractor.extract(imagen0)
        feats1 = extractor.extract(imagen1)
        
        # Emparejar características del primer conjunto
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        
        # Obtener los puntos clave y las correspondencias del primer conjunto
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
        # Filtrar las correspondencias por score
        scores = matches01["scores"]
        indices_filtrados = scores >= umbral_score
        m_kpts0_filtrados = m_kpts0[indices_filtrados]
        m_kpts1_filtrados = m_kpts1[indices_filtrados]
        scores_filtrados = scores[indices_filtrados]
        
        # Convertir a numpy para calcular la homografía
        pts0 = m_kpts0_filtrados.cpu().numpy()
        pts1 = m_kpts1_filtrados.cpu().numpy()
        
        # Calcular la matriz de homografía con RANSAC y obtener la máscara de inliers
        M, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        
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
        imagen0_warped_pil = Image.fromarray(imagen0_warped)
        
        # Convertir la imagen warpada a tensor
        imagen0_warped_tensor = torch.tensor(np.array(imagen0_warped_pil)).permute(2, 0, 1).float() / 255.0
        imagen0_warped_tensor = imagen0_warped_tensor.to(dispositivo)
        
        # Extraer características de la imagen warpada y de la segunda imagen
        feats0_warped = extractor.extract(imagen0_warped_tensor)
        feats1_warped = extractor.extract(imagen1)
        
        # Emparejar características nuevamente
        matches01_warped = matcher({'image0': feats0_warped, 'image1': feats1_warped})
        feats0_warped, feats1_warped, matches01_warped = [rbd(x) for x in [feats0_warped, feats1_warped, matches01_warped]]
        
        # Obtener los puntos clave y las correspondencias del segundo conjunto
        kpts0_warped, kpts1_warped, matches_warped = feats0_warped["keypoints"], feats1_warped["keypoints"], matches01_warped["matches"]
        m_kpts0_warped, m_kpts1_warped = kpts0_warped[matches_warped[..., 0]], kpts1_warped[matches_warped[..., 1]]
        
        # Filtrar las correspondencias por score del segundo conjunto
        scores_warped = matches01_warped["scores"]
        indices_filtrados_warped = scores_warped >= umbral_score
        m_kpts0_warped_filtrados = m_kpts0_warped[indices_filtrados_warped]
        m_kpts1_warped_filtrados = m_kpts1_warped[indices_filtrados_warped]
        scores_filtrados_warped = scores_warped[indices_filtrados_warped]
        
        # Crear una imagen combinada horizontalmente
        imagen0_warped_np = imagen0_warped_tensor.cpu().permute(1, 2, 0).numpy()
        imagen1_np = imagen1.cpu().permute(1, 2, 0).numpy()
        combined_image = np.hstack((imagen0_warped_np, imagen1_np))
        
        
        plt.imshow(imagen0_warped_np)

        fusioned_image = cv2.add(imagen0_warped_np, imagen1_np)

        # Plotear la imagen combinada
        plt.figure(figsize=(20, 10))
        plt.imshow(fusioned_image)
        
        # Calcular métricas para el primer conjunto de correspondencias
        total_matches = len(m_kpts0_filtrados)
        inliers = mask.sum()
        outliers = total_matches - inliers
        inlier_ratio = inliers / total_matches if total_matches > 0 else 0
        distancia_media = calcular_distancia_media(pts0, pts1)
        error_reproyeccion = evaluar_homografia(M, pts0, pts1)
        
        # Dibujar las correspondencias del segundo conjunto (post warping)
        if mostrar_correspondencias:
            for i in range(len(m_kpts0_warped_filtrados)):
                # Determinar el color basado en el score
                score_tmp = scores_filtrados_warped[i].item()
                if score_tmp >= 0.90:
                    color = 'g'  # Verde
                elif 0.75 <= score_tmp < 0.90:
                    color = '#FFFF00'  # Amarillo
                elif 0.60 <= score_tmp < 0.75:
                    color = '#FFA500'  # Naranja
                elif 0.50 <= score_tmp < 0.60:
                    color = 'r'  # Rojo
                else:
                    continue  # Ignorar correspondencias con score < 0.50
                
                # Obtener las coordenadas en la imagen combinada
                x0_warped, y0_warped = m_kpts0_warped_filtrados[i].cpu().numpy()
                x1_warped, y1_warped = m_kpts1_warped_filtrados[i].cpu().numpy()
                x1_shifted_warped = x1_warped + imagen0_warped_np.shape[1]  # Ajustar la posición horizontal de la segunda imagen
                
                # Dibujar línea entre los puntos correspondientes
                plt.plot([x0_warped, x1_shifted_warped], [y0_warped, y1_warped], c=color, linewidth=0.5)
                
                if score_tmp >= 0.50:
                    # Dibujar puntos en la imagen combinada
                    plt.scatter(x0_warped, y0_warped, c='b', s=10)
                    plt.scatter(x1_shifted_warped, y1_warped, c='b', s=10)
        
        # Añadir texto con métricas si está habilitado
        if mostrar_metricas:
            plt.text(10, 30, f"Total Matches: {total_matches}", color='white', fontsize=12, 
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 50, f"Inliers: {int(inliers)}", color='green', fontsize=12, 
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 70, f"Outliers: {int(outliers)}", color='red', fontsize=12, 
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 90, f"Inlier Ratio: {inlier_ratio:.2f}", color='yellow', fontsize=12, 
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 110, f"Mean Euclidean Distance: {distancia_media:.2f} pixels", color='white', fontsize=12, 
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 130, f"Mean Reprojection Error: {error_reproyeccion:.2f} pixels", color='white', fontsize=12, 
                     bbox=dict(facecolor='black', alpha=0.5))
        
        plt.axis('off')
        plt.title('Correspondencias entre keypoints después de la transformación de perspectiva', fontsize=20, fontweight="bold")
        
        if guardar_figura:
            # Asegurar que el directorio existe
            ruta_directorio = os.path.dirname(ruta_guardado)
            if ruta_directorio:
                os.makedirs(ruta_directorio, exist_ok=True)
            plt.savefig(ruta_guardado, bbox_inches='tight', pad_inches=0)
            print(f"Figura guardada en {ruta_guardado}")
        
        plt.show()
        plt.close()
        
        if mostrar_metricas:
            # Mostrar histograma de scores
            plt.figure(figsize=(10, 6))
            sns.histplot(scores_filtrados_warped.cpu().detach().numpy(), bins=50, kde=True)
            plt.title('Distribución de Scores de Correspondencia (Post Warping)')
            plt.xlabel('Score')
            plt.ylabel('Frecuencia')
            plt.show()

            # Mostrar métricas en la consola
            print(f"Total Matches: {total_matches}")
            print(f"Inliers: {inliers}")
            print(f"Outliers: {outliers}")
            print(f"Inlier Ratio: {inlier_ratio:.2f}")
            print(f"Mean Euclidean Distance: {distancia_media:.2f} pixels")
            print(f"Mean Reprojection Error: {error_reproyeccion:.2f} pixels")

    except Exception as e:
            print(f"Error al procesar y visualizar las correspondencias: {e}")

        
if __name__ == "__main__":
    # Ruta de las imagenes 
    # SAME RESOLUTION ON RAW IMAGES
    img_0 = "Cameras/captures/visible/left/LEFT_visible_20241107_163226.png"
    img_1 = "Cameras/captures/thermal/thermal_20241107_163226.png"

    # # NOT SAME RESOLUTION ON RAW IMAGES
    # img_0 = "captures/rectified/left/LEFT_visible_20241015_153216.png"
    # img_1 = "captures/thermal/thermal_20241015_153216.png"


    # THRESHOLD TESTING
    img_result = procesar_imagenes(ruta_imagen0=img_0, ruta_imagen1=img_1, threshold=200)
    imagen0_warped_bgr = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)


    # MOSTRAR IMAGEN WARPADA
    # cv2.imshow('Imagen Warpada', imagen0_warped_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    

    # MOSTAR IMAGEN ORIGINAL TERMICA
    # img_1 = cv2.imread(img_1, cv2.IMREAD_GRAYSCALE)
    # # img_1 = cv2.applyColorMap(img_1, cv2.COLORMAP_JET)
    # cv2.imshow('Imagen Termal', img_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 


    # # MOSTRAR IMAGEN FUSIONADA PARA VER DIFERENCIAS
    # img_1 = cv2.imread(img_1, cv2.IMREAD_GRAYSCALE)
    # # Esto es solo para poder funsioanr ambas imagenes
    # img_1_rgb = cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR)
    
    # if imagen0_warped_bgr.shape != img_1_rgb.shape:
    #     print("Las resoluciones no son iguales. \nimg0:{imagen0_warped_bgr.shape}\nimg1:{img_1_rgb.shape}")
        
        
    #     img_1_resized = cv2.resize(img_1_rgb, (imagen0_warped_bgr.shape[1], imagen0_warped_bgr.shape[0]))

    # fusioned_image = cv2.add(imagen0_warped_bgr, img_1_rgb)
    # # img_1 = cv2.applyColorMap(img_1, cv2.COLORMAP_JET)
    # cv2.imshow('Register Differences', fusioned_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 


    # Leer la imagen original en escala de grises
    img_1 = cv2.imread(img_1, cv2.IMREAD_GRAYSCALE)

    # Convertir img_1 a una imagen RGB para poder modificar los canales
    img_1_rgb = cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR)

    # Redimensionar img_1_rgb para que coincida con las dimensiones de imagen0_warped_bgr
    img_1_resized = cv2.resize(img_1_rgb, (imagen0_warped_bgr.shape[1], imagen0_warped_bgr.shape[0]))

    # Crear una imagen con solo el canal rojo de img_1
    img_1_red = img_1_resized.copy()
    img_1_red[:, :, 0] = 0  # Poner el canal azul a 0
    img_1_red[:, :, 1] = 0  # Poner el canal verde a 0

    # Asegurar que imagen0_warped_bgr tenga solo el canal azul
    imagen0_warped_blue = imagen0_warped_bgr.copy()
    imagen0_warped_blue[:, :, 1] = 0  # Poner el canal verde a 0
    imagen0_warped_blue[:, :, 2] = 0  # Poner el canal rojo a 0

    # Fusionar ambas imágenes
    fusioned_image = cv2.add(img_1_red, imagen0_warped_blue)

    # Mostrar la imagen fusionada
    cv2.imshow('Register Differences', fusioned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # # Directorios que contienen las imágenes
    # path_dir0 = "../captures/rectified/left/"
    # path_dir1 = "../captures/thermal/"
    
    # # Extensiones de imagen a considerar
    # img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']

    # # Obtener lista de imágenes en cada directorio
    # images0 = []
    # for ext in img_extensions:
    #     images0.extend(glob.glob(os.path.join(path_dir0, ext)))

    # images1 = []
    # for ext in img_extensions:
    #     images1.extend(glob.glob(os.path.join(path_dir1, ext)))

    # # Ordenar las listas de imágenes
    # images0.sort()
    # images1.sort()

    # # Verificar que ambas listas tengan imágenes
    # if not images0:
    #     print(f"No se encontraron imágenes en el directorio: {path_dir0}")
    #     exit(1)
    # if not images1:
    #     print(f"No se encontraron imágenes en el directorio: {path_dir1}")
    #     exit(1)

    # # Determinar la cantidad mínima de imágenes para evitar errores
    # len0 = len(images0)
    # len1 = len(images1)
    # min_len = min(len0, len1)

    # print(f"Total imágenes en {path_dir0}: {len0}")
    # print(f"Total imágenes en {path_dir1}: {len1}")
    # print(f"Procesando {min_len} pares de imágenes...")

    # # Inicializar índice para el ciclo while
    # index = 0

    # while index < min_len:
    #     ruta0 = images0[index]
    #     ruta1 = images1[index]

    #     # Definir filtros para cada imagen (puedes modificar estos valores según tus necesidades)
    #     filtro0 = "none"  # Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none"
    #     filtro1 = "none"  # Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none"

    #     # Generar un nombre único para guardar las correspondencias (usando el índice)
    #     nombre_guardado = f"correspondencias_{index + 1}.png"
    #     ruta_guardado = os.path.join("resultados", nombre_guardado)

    #     print(f"\nProcesando par {index + 1}:")
    #     print(f"Imagen0: {ruta0}")
    #     print(f"Imagen1: {ruta1}")
    #     print(f"Guardando correspondencias en: {ruta_guardado}")

    #     # Llamar a la función para mostrar correspondencias mejoradas
    #     mostrar_correspondencias_mejorada(
    #         ruta_imagen0=ruta0,
    #         ruta_imagen1=ruta1,
    #         extractor_tipo="superpoint",
    #         max_num_keypoints=2048,
    #         umbral_score=0.5,
    #         guardar_figura=True,
    #         ruta_guardado=ruta_guardado,
    #         filtro_imagen0=filtro0,       # Aplicar filtro a imagen0
    #         filtro_imagen1=filtro1,       # Aplicar filtro a imagen1
    #         mostrar_correspondencias=False, # Activar/desactivar líneas de correspondencia
    #         mostrar_metricas=True          # Mostrar/ocultar métricas
    #     )

    #     index += 1

    # print("\nProcesamiento completado.")