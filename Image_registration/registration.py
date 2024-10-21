import torch
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet, viz2d
from lightglue.utils import load_image, rbd
from torchvision.transforms.functional import resize, to_pil_image
from torchvision.utils import save_image
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def procesar_imagenes(
    ruta_imagen1,
    ruta_imagen2,
    extractor_tipo="superpoint",
    max_num_keypoints=4096,
    dispositivo=None
):
    """
    Procesa dos imágenes para obtener una imagen resultante tras aplicar la transformación de perspectiva.

    Args:
        ruta_imagen1 (str): Ruta al primer archivo de imagen.
        ruta_imagen2 (str): Ruta al segundo archivo de imagen.
        extractor_tipo (str, opcional): Tipo de extractor a utilizar. Opciones: "superpoint", "disk", "sift", "aliked", "doghardnet". Por defecto es "superpoint".
        max_num_keypoints (int, opcional): Número máximo de puntos clave a extraer. Por defecto es 4096.
        dispositivo (torch.device, opcional): Dispositivo para ejecutar el modelo. Si es None, se selecciona automáticamente GPU si está disponible.

    Returns:
        PIL.Image: Imagen resultante tras la transformación de perspectiva.
    """
    
    # Configurar el dispositivo
    if dispositivo is None:
        dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Dispositivo utilizado:", dispositivo)
    
    # Inicializar extractor y matcher según el tipo especificado
    if extractor_tipo == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='superpoint').eval().to(dispositivo)
    elif extractor_tipo == "disk":
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='disk').eval().to(dispositivo)
    elif extractor_tipo == "sift":
        extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='sift').eval().to(dispositivo)
    elif extractor_tipo == "aliked":
        extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='aliked').eval().to(dispositivo)
    elif extractor_tipo == "doghardnet":
        extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='doghardnet').eval().to(dispositivo)
    else:
        raise ValueError(f"Extractor tipo '{extractor_tipo}' no es válido. Selecciona entre 'superpoint', 'disk', 'sift', 'aliked', 'doghardnet'.")

    try:
        # Cargar y preparar las imágenes
        imagen0 = load_image(ruta_imagen1).to(dispositivo)
        imagen1 = load_image(ruta_imagen2).to(dispositivo)
        
        # Extraer características
        feats0 = extractor.extract(imagen0)
        feats1 = extractor.extract(imagen1)
        
        # Emparejar características
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        
        matches, scores = matches01["matches"], matches01["scores"]
        points0 = feats0['keypoints'][matches[..., 0]]
        points1 = feats1['keypoints'][matches[..., 1]]
        
        pts0 = points0.cpu().numpy()
        pts1 = points1.cpu().numpy()
        
        # Calcular la matriz de homografía
        M, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        
        if M is None:
            raise ValueError("No se pudo calcular la homografía.")
        
        # Convertir tensor de PyTorch a imagen PIL para la transformación
        imagen0_pil = to_pil_image(imagen0)
        imagen1_pil = to_pil_image(imagen1)
        
        # Aplicar la transformación de perspectiva
        imagen0_warped = cv2.warpPerspective(
            np.array(imagen0_pil), 
            M, 
            (imagen1.shape[2], imagen1.shape[1])
        )
        warped_image0 = torch.tensor(imagen0_warped).permute(2, 0, 1)
        
        # Convertir de nuevo a PIL y normalizar
        warped_image0_pil = to_pil_image(warped_image0)
        warped_image0_tensor = torch.tensor(np.array(warped_image0_pil)).permute(2, 0, 1).float() / 255.0
        warped_image0_tensor = resize(warped_image0_tensor, (480, 640))
        
        # Convertir a PIL para retornar
        resultado_pil = to_pil_image(warped_image0_tensor)
        
        return resultado_pil

    except Exception as e:
        print(f"Error al procesar las imágenes: {e}")
        return None

def mostrar_correspondencias(
    ruta_imagen1,
    ruta_imagen2,
    extractor_tipo="superpoint",
    max_num_keypoints=4096,
    dispositivo=None,
    umbral_score=0.5,
    guardar_figura=True,
    ruta_guardado="correspondencias"
):
    """
    Muestra la correspondencia de los puntos clave entre dos imágenes.

    Args:
        ruta_imagen1 (str): Ruta al primer archivo de imagen.
        ruta_imagen2 (str): Ruta al segundo archivo de imagen.
        extractor_tipo (str, opcional): Tipo de extractor a utilizar. Opciones: "superpoint", "disk", "sift", "aliked", "doghardnet". Por defecto es "superpoint".
        max_num_keypoints (int, opcional): Número máximo de puntos clave a extraer. Por defecto es 4096.
        dispositivo (torch.device, opcional): Dispositivo para ejecutar el modelo. Si es None, se selecciona automáticamente GPU si está disponible.
        umbral_score (float, opcional): Umbral de score para filtrar las correspondencias antes de visualizarlas. Valores entre 0 y 1. Por defecto es 0.5.
        guardar_figura (bool, opcional): Si se establece en True, guarda la figura en la ruta especificada. Por defecto es False.
        ruta_guardado (str, opcional): Ruta donde se guardará la figura si `guardar_figura` es True. Por defecto es "correspondencias.png".

    Returns:
        None. Muestra una figura con las correspondencias entre los puntos clave de las dos imágenes.
    """
    
    # Configurar el dispositivo
    if dispositivo is None:
        dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Dispositivo utilizado:", dispositivo)
    
    # Inicializar extractor y matcher según el tipo especificado
    if extractor_tipo == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='superpoint').eval().to(dispositivo)
    elif extractor_tipo == "disk":
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='disk').eval().to(dispositivo)
    elif extractor_tipo == "sift":
        extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='sift').eval().to(dispositivo)
    elif extractor_tipo == "aliked":
        extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='aliked').eval().to(dispositivo)
    elif extractor_tipo == "doghardnet":
        extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='doghardnet').eval().to(dispositivo)
    else:
        raise ValueError(f"Extractor tipo '{extractor_tipo}' no es válido. Selecciona entre 'superpoint', 'disk', 'sift', 'aliked', 'doghardnet'.")
    
    try:
        # Verificar que las rutas sean cadenas de texto
        if not isinstance(ruta_imagen1, (str, os.PathLike)) or not isinstance(ruta_imagen2, (str, os.PathLike)):
            raise TypeError("Las rutas de las imágenes deben ser cadenas de texto, bytes o objetos Path.")
        
        # Verificar que las rutas existan
        if not os.path.exists(ruta_imagen1):
            raise FileNotFoundError(f"La ruta de la imagen1 no existe: {ruta_imagen1}")
        if not os.path.exists(ruta_imagen2):
            raise FileNotFoundError(f"La ruta de la imagen2 no existe: {ruta_imagen2}")
        
        # Cargar y preparar las imágenes
        imagen0 = load_image(ruta_imagen1).to(dispositivo)
        imagen1 = load_image(ruta_imagen2).to(dispositivo)
        
        # Extraer características
        feats0 = extractor.extract(imagen0)
        feats1 = extractor.extract(imagen1)
        
        # Emparejar características
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        
        # Obtener los puntos clave y las correspondencias
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
        # Filtrar las correspondencias por score
        scores = matches01["scores"]
        indices_filtrados = scores >= umbral_score
        m_kpts0_filtrados = m_kpts0[indices_filtrados]
        m_kpts1_filtrados = m_kpts1[indices_filtrados]
        scores_filtrados = scores[indices_filtrados]
        
        # Visualización de las correspondencias
        viz2d.plot_images([imagen0, imagen1])  # Sin pasar 'ax'
        ax = plt.gca()  # Obtener el eje actual
        
        # Dibujar las correspondencias con diferentes colores según el score
        for i in range(len(m_kpts0_filtrados)):
            score_tmp = scores_filtrados[i].item()
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
            
            # Dibujar línea entre los puntos correspondientes
            ax.plot(
                [m_kpts0_filtrados[i, 0], m_kpts1_filtrados[i, 0] + imagen0.shape[2]], 
                [m_kpts0_filtrados[i, 1], m_kpts1_filtrados[i, 1]], 
                c=color, linewidth=0.5
            )
            
            if score_tmp >= 0.50:
                # Dibujar puntos en la imagen combinada
                ax.scatter(m_kpts0_filtrados[i, 0], m_kpts0_filtrados[i, 1], c='b', s=10)
                ax.scatter(m_kpts1_filtrados[i, 0] + imagen0.shape[2], m_kpts1_filtrados[i, 1], c='b', s=10)
        
        ax.axis('off')
        ax.set_title('Correspondencias entre puntos clave')
        
        if guardar_figura:
            # Asegurar que el directorio existe
            ruta_directorio = os.path.dirname(ruta_guardado)
            if ruta_directorio:
                os.makedirs(ruta_directorio, exist_ok=True)
            plt.savefig(ruta_guardado, bbox_inches='tight', pad_inches=0)
            print(f"Figura guardada en {ruta_guardado}")
        
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Error al procesar y visualizar las correspondencias: {e}")

def mostrar_correspondencias_mejorada(
    ruta_imagen1,
    ruta_imagen2,
    extractor_tipo="superpoint",
    max_num_keypoints=4096,
    dispositivo=None,
    umbral_score=0.5,
    guardar_figura=False,
    ruta_guardado="correspondencias_mejoradas.png"
):
    """
    Muestra la correspondencia de los puntos clave entre dos imágenes después de aplicar una transformación de perspectiva.

    Args:
        ruta_imagen1 (str): Ruta al primer archivo de imagen.
        ruta_imagen2 (str): Ruta al segundo archivo de imagen.
        extractor_tipo (str, opcional): Tipo de extractor a utilizar. Opciones: "superpoint", "disk", "sift", "aliked", "doghardnet". Por defecto es "superpoint".
        max_num_keypoints (int, opcional): Número máximo de puntos clave a extraer. Por defecto es 4096.
        dispositivo (torch.device, opcional): Dispositivo para ejecutar el modelo. Si es None, se selecciona automáticamente GPU si está disponible.
        umbral_score (float, opcional): Umbral de score para filtrar las correspondencias antes de visualizarlas. Valores entre 0 y 1. Por defecto es 0.5.
        guardar_figura (bool, opcional): Si se establece en True, guarda la figura en la ruta especificada. Por defecto es False.
        ruta_guardado (str, opcional): Ruta donde se guardará la figura si `guardar_figura` es True. Por defecto es "correspondencias_mejoradas.png".

    Returns:
        None. Muestra una figura con las correspondencias entre los puntos clave de las dos imágenes.
    """
    
    # Configurar el dispositivo
    if dispositivo is None:
        dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Dispositivo utilizado:", dispositivo)
    
    # Inicializar extractor y matcher según el tipo especificado
    if extractor_tipo == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='superpoint').eval().to(dispositivo)
    elif extractor_tipo == "disk":
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='disk').eval().to(dispositivo)
    elif extractor_tipo == "sift":
        extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='sift').eval().to(dispositivo)
    elif extractor_tipo == "aliked":
        extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='aliked').eval().to(dispositivo)
    elif extractor_tipo == "doghardnet":
        extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(dispositivo)
        matcher = LightGlue(features='doghardnet').eval().to(dispositivo)
    else:
        raise ValueError(f"Extractor tipo '{extractor_tipo}' no es válido. Selecciona entre 'superpoint', 'disk', 'sift', 'aliked', 'doghardnet'.")
    
    try:
        # Verificar que las rutas sean cadenas de texto
        if not isinstance(ruta_imagen1, (str, os.PathLike)) or not isinstance(ruta_imagen2, (str, os.PathLike)):
            raise TypeError("Las rutas de las imágenes deben ser cadenas de texto, bytes o objetos Path.")
        
        # Verificar que las rutas existan
        if not os.path.exists(ruta_imagen1):
            raise FileNotFoundError(f"La ruta de la imagen1 no existe: {ruta_imagen1}")
        if not os.path.exists(ruta_imagen2):
            raise FileNotFoundError(f"La ruta de la imagen2 no existe: {ruta_imagen2}")
        
        # Cargar y preparar las imágenes
        imagen0 = load_image(ruta_imagen1).to(dispositivo)
        imagen1 = load_image(ruta_imagen2).to(dispositivo)
        
        # Extraer características
        feats0 = extractor.extract(imagen0)
        feats1 = extractor.extract(imagen1)
        
        # Emparejar características
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        
        # Obtener los puntos clave y las correspondencias
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
        
        # Calcular la matriz de homografía
        M, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        
        if M is None:
            raise ValueError("No se pudo calcular la homografía.")
        
        # Convertir tensor de PyTorch a imagen PIL para la transformación
        imagen0_pil = to_pil_image(imagen0)
        imagen1_pil = to_pil_image(imagen1)
        
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
        feats1 = extractor.extract(imagen1)
        
        # Emparejar características nuevamente
        matches01_warped = matcher({'image0': feats0_warped, 'image1': feats1})
        feats0_warped, feats1, matches01_warped = [rbd(x) for x in [feats0_warped, feats1, matches01_warped]]
        
        # Obtener los puntos clave y las correspondencias
        kpts0_warped, kpts1, matches_warped = feats0_warped["keypoints"], feats1["keypoints"], matches01_warped["matches"]
        m_kpts0_warped, m_kpts1 = kpts0_warped[matches_warped[..., 0]], kpts1[matches_warped[..., 1]]
        
        # Filtrar las correspondencias por score
        scores_warped = matches01_warped["scores"]
        indices_filtrados_warped = scores_warped >= umbral_score
        m_kpts0_warped_filtrados = m_kpts0_warped[indices_filtrados_warped]
        m_kpts1_filtrados = m_kpts1[indices_filtrados_warped]
        scores_filtrados_warped = scores_warped[indices_filtrados_warped]
        
        # Crear una imagen combinada horizontalmente
        imagen0_warped_np = imagen0_warped_tensor.cpu().permute(1, 2, 0).numpy()
        imagen1_np = imagen1.cpu().permute(1, 2, 0).numpy()
        combined_image = np.hstack((imagen0_warped_np, imagen1_np))
        
        # Plotear la imagen combinada
        plt.figure(figsize=(20, 10))
        plt.imshow(combined_image)
        
        # Dibujar las correspondencias con diferentes colores según el score
        for i in range(len(m_kpts0_warped_filtrados)):
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
            
            # Coordenadas en la imagen combinada
            x0, y0 = m_kpts0_warped_filtrados[i]
            x1, y1 = m_kpts1_filtrados[i]
            x1_shifted = x1 + imagen0_warped_np.shape[1]  # Ajustar la posición horizontal de la segunda imagen
            
            # Dibujar línea entre los puntos correspondientes
            plt.plot([x0, x1_shifted], [y0, y1], c=color, linewidth=0.5)
            
            if score_tmp >= 0.50:
                # Dibujar puntos en la imagen combinada
                plt.scatter(x0, y0, c='b', s=10)
                plt.scatter(x1_shifted, y1, c='b', s=10)
        
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
    except Exception as e:
        print(f"Error al procesar y visualizar las correspondencias: {e}")

if __name__ == "__main__":
    ruta1="../captures/visible/left/LEFT_visible_20241015_120122.png"
    ruta2="../captures/thermal/thermal_20241015_120122.png"
    
    # resultado = procesar_imagenes(
    #     ruta_imagen1=ruta1,
    #     ruta_imagen2=ruta2,
    #     extractor_tipo="superpoint"
    # )
    # if resultado:
    #     resultado.show()


    mostrar_correspondencias_mejorada(
        ruta_imagen1=ruta1,
        ruta_imagen2=ruta2,
        extractor_tipo="superpoint",
        max_num_keypoints=2048,
        umbral_score=0.5,
        guardar_figura=False,
        ruta_guardado="resultados/correspondencias.png"
    )
