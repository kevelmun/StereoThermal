import cv2
from util import *
from Image_registration import registration
import cv2
from util import show_image
from Image_registration import registration

import glob
import os
import matplotlib.pyplot as plt
# Función para cargar las imágenes desde un directorio
def cargar_imagenes(path_dir, img_extensions):
    """
    Carga las imágenes de un directorio dado, considerando las extensiones de imagen especificadas.

    :param path_dir: Directorio de imágenes.
    :param img_extensions: Lista de extensiones de archivo a buscar (por ejemplo, '*.png', '*.jpg').
    :return: Lista de rutas de las imágenes encontradas.
    """
    imagenes = []
    for ext in img_extensions:
        imagenes.extend(glob.glob(os.path.join(path_dir, ext)))
    
    imagenes.sort()  # Ordenar las imágenes alfabéticamente
    return imagenes

# Función para procesar los pares de imágenes
def procesar_correspondencias(path_dir0, path_dir1, img_extensions, filtro0, filtro1):
    """
    Procesa las imágenes de dos directorios, generando las correspondencias entre ellas.

    :param path_dir0: Directorio de imágenes de la primera serie.
    :param path_dir1: Directorio de imágenes de la segunda serie.
    :param img_extensions: Lista de extensiones de archivo a considerar.
    :params filtro0, filtro1: Filtros para cada imagen - Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none"
    """
    

    # Cargar las imágenes desde ambos directorios
    images0 = cargar_imagenes(path_dir0, img_extensions)
    images1 = cargar_imagenes(path_dir1, img_extensions)

    # Verificar que ambas listas tengan imágenes
    if not images0:
        print(f"No se encontraron imágenes en el directorio: {path_dir0}")
        return
    if not images1:
        print(f"No se encontraron imágenes en el directorio: {path_dir1}")
        return

    # Determinar la cantidad mínima de imágenes para evitar errores
    min_len = min(len(images0), len(images1))

    print(f"Total imágenes en {path_dir0}: {len(images0)}")
    print(f"Total imágenes en {path_dir1}: {len(images1)}")
    print(f"Procesando {min_len} pares de imágenes...")

    # Procesar cada par de imágenes
    for index in range(min_len):
        ruta0 = images0[index]
        ruta1 = images1[index]

        

        # Generar un nombre único para guardar las correspondencias (usando el índice)
        nombre_guardado = f"correspondencias_{index + 1}.png"
        ruta_guardado = os.path.join("resultados", nombre_guardado)

        print(f"\nProcesando par {index + 1}:")
        print(f"Imagen0: {ruta0}")
        print(f"Imagen1: {ruta1}")
        print(f"Guardando correspondencias en: {ruta_guardado}")

        # Llamar a la función para mostrar correspondencias mejoradas
        registration.mostrar_correspondencias_mejorada(
            ruta_imagen0=ruta0,
            ruta_imagen1=ruta1,
            extractor_tipo="superpoint",
            max_num_keypoints=2048,
            umbral_score=0.5,
            guardar_figura=True,
            ruta_guardado=ruta_guardado,
            filtro_imagen0=filtro0,       # Aplicar filtro a imagen0
            filtro_imagen1=filtro1,       # Aplicar filtro a imagen1
            mostrar_correspondencias=False, # Activar/desactivar líneas de correspondencia
            mostrar_metricas=True          # Mostrar/ocultar métricas
        )

    print("\nProcesamiento completado.")


def fusion_images(img_1_path, imagen0_warped_bgr):
    """
    Fusiona una imagen de entrada en escala de grises con una imagen RGB modificada,
    y muestra la imagen fusionada.

    :param img_1_path: Ruta de la imagen en escala de grises.
    :param imagen0_warped_bgr: Imagen en formato BGR (generalmente la imagen base).
    :return: La imagen fusionada (BGR).
    """
    # Leer la imagen original en escala de grises
    img_1 = cv2.imread(img_1_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img_1 is None:
        print(f"Error al cargar la imagen: {img_1_path}")
        return None

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
    show_image(fusioned_image, 'Register Differences')
    return fusioned_image

def merge_and_display_differences(image_0: np.ndarray, image_1: np.ndarray):
    """Fusiona dos imágenes y muestra las diferencias."""
    # Convertir la imagen térmica a RGB para que coincida en canales
    image_1_bgr = cv2.cvtColor(image_1, cv2.COLOR_GRAY2BGR)
    
    # Asegurarse de que ambas imágenes tengan el mismo tamaño
    if image_0.shape != image_1_bgr.shape:
        print(f"Las resoluciones no son iguales.\nimage_0: {image_0.shape}\nimage_1: {image_1_bgr.shape}")
        image_1_bgr = cv2.resize(image_1_bgr, (image_0.shape[1], image_0.shape[0]))

    # Fusionar las imágenes
    merged_image = cv2.add(image_0, image_1_bgr)
    
    # Mostrar la imagen fusionada
    show_image(merged_image, 'Registered Differences')


def main():
    # Rutas de las imágenes
    image_0_path = "Cameras/captures/thermal/thermal_20241107_163226.png"
    image_1_path = "Cameras/captures/visible/left/LEFT_visible_20241107_163226.png"
    matlab_warped = "Cameras/captures/matlab_registration/output_image2.png"
    matlab_warped = cv2.imread(matlab_warped, cv2.IMREAD_GRAYSCALE)
    # Umbral para el registro
    threshold = 200
    
    
    # Procesar las imágenes utilizando el registro
    image_warped, matches, reliable_points, error = registration.procesar_imagenes(ruta_imagen0=image_0_path, ruta_imagen1=image_1_path, threshold=threshold)
    
            
    # reliable_points = []
    # for i in range(len(points0)):
    #     score_tmp = scores[i].item()
    #     if score_tmp >= 0.75:
    #         reliable_points.add(scores[i])
    
    # Mostrar la imagen warpada
    # show_image(image_warped, 'Warped Image')
    
    # # Mostrar la imagen térmica original
    # image_0_grayscale = cv2.imread(image_0_path, cv2.IMREAD_GRAYSCALE)
    # show_image(image_0_grayscale, 'Thermal Image')

    # # Fusionar las imágenes y mostrar las diferencias
    image_1 = cv2.imread(image_1_path, cv2.IMREAD_COLOR)
    
    # SI USAS REGISTRADO PREVIO CON MATLAB COMENTA ESTO
    image_warped = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY)


    merge_and_display_differences(image_1, image_warped)

    # fusioned_image = fusion_images(image_1_path, image_warped)


    # # Directorios que contienen las imágenes
    # path_dir0 = "../captures/rectified/left/"
    # path_dir1 = "../captures/thermal/"

    # # Extensiones de imagen a considerar
    # img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']

    # # Llamar a la función principal de procesamiento
    # procesar_correspondencias(path_dir0, path_dir1, img_extensions)


if __name__ == "__main__":
    # main()
    fixed_image_path = "Cameras/captures/visible/left/LEFT_visible_20241107_163226.png"
    
    moving_image_path = "Cameras/captures/thermal/THERMAL_20241107_163226.png"

    # Load the image in grayscale
    image = cv2.imread(fixed_image_path, 1)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)

    # Display the original and equalized images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Equalized Image')
    plt.imshow(equalized_image, cmap='gray')

    plt.show()

    # # Load the images
    # fixed_image = cv2.imread(fixed_image_path, 1)
    # moving_image = cv2.imread(moving_image_path, 1)

    # # Detect ORB features and compute descriptors
    # orb = cv2.ORB_create()
    # keypoints1, descriptors1 = orb.detectAndCompute(fixed_image, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(moving_image, None)

    # # Match features using the BFMatcher
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(descriptors1, descriptors2)
    # matches = sorted(matches, key=lambda x: x.distance)

    # # Extract location of good matches
    # points1 = np.zeros((len(matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(matches), 2), dtype=np.float32)

    # for i, match in enumerate(matches):
    #     points1[i, :] = keypoints1[match.queryIdx].pt
    #     points2[i, :] = keypoints2[match.trainIdx].pt

    # # Estimate the transformation matrix
    # matrix, mask = cv2.estimateAffinePartial2D(points2, points1)

    # # Apply the transformation to the moving image
    # aligned_image = cv2.warpAffine(moving_image, matrix, (fixed_image.shape[1], fixed_image.shape[0]))
    
    # fusion_images(fixed_image_path, aligned_image)



