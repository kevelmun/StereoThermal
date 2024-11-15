import cv2
import numpy as np
import glob
import os
# Parámetros iniciales
CHECKERBOARD_SIZE = (6, 9)  # Ajustar según tu patrón
square_size = 20.0  # Tamaño de los cuadrados del checkerboard en mm

# Criterios para cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepara puntos del mundo (coordenadas 3D reales del patrón)
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * square_size

# Almacenar puntos 3D y puntos 2D de las imágenes
objpoints_stereo_color = []  # Puntos 3D para calibración estéreo color
imgpoints_color1_stereo = []  # Puntos 2D de la cámara color 1 para estéreo
imgpoints_color2_stereo = []  # Puntos 2D de la cámara color 2 para estéreo

objpoints_stereo_thermal = []  # Puntos 3D para calibración estéreo térmica
imgpoints_color1_thermal = []  # Puntos 2D de la cámara color 1 para estéreo térmico
imgpoints_thermal = []  # Puntos 2D de la cámara térmica

# Para calibración individual
objpoints_color1 = []
imgpoints_color1 = []

objpoints_color2 = []
imgpoints_color2 = []

objpoints_thermal_individual = []
imgpoints_thermal_individual = []

# Carga las imágenes para cada cámara
color1_images = glob.glob('../CalibrationData/Steven/left/*.png')
color2_images = glob.glob('../CalibrationData/Steven/right/*.png')
thermal_images = glob.glob('../CalibrationData/Steven/thermal_invert/*.png')

# Asegúrate de que todas las listas de imágenes estén ordenadas y tengan la misma longitud
color1_images.sort()
color2_images.sort()
thermal_images.sort()

num_images = min(len(color1_images), len(color2_images), len(thermal_images))

# Detectar esquinas del checkerboard y sincronizar detecciones
for i in range(num_images):
    color1_img_path = color1_images[i]
    color2_img_path = color2_images[i]
    thermal_img_path = thermal_images[i]

    # Leer imágenes
    img_color1 = cv2.imread(color1_img_path, cv2.IMREAD_GRAYSCALE)
    img_color2 = cv2.imread(color2_img_path, cv2.IMREAD_GRAYSCALE)
    img_thermal = cv2.imread(thermal_img_path, cv2.IMREAD_GRAYSCALE)

    # Detectar esquinas en cámara color 1
    ret1, corners1 = cv2.findChessboardCorners(img_color1, CHECKERBOARD_SIZE, None)
    if ret1:
        corners_refined1 = cv2.cornerSubPix(img_color1, corners1, (11, 11), (-1, -1), criteria)
        imgpoints_color1.append(corners_refined1)
        objpoints_color1.append(objp)

    # Detectar esquinas en cámara color 2
    ret2, corners2 = cv2.findChessboardCorners(img_color2, CHECKERBOARD_SIZE, None)
    if ret2:
        corners_refined2 = cv2.cornerSubPix(img_color2, corners2, (11, 11), (-1, -1), criteria)
        imgpoints_color2.append(corners_refined2)
        objpoints_color2.append(objp)

    # Detectar esquinas en cámara térmica
    ret3, corners3 = cv2.findChessboardCorners(img_thermal, CHECKERBOARD_SIZE, None)
    if ret3:
        corners_refined3 = cv2.cornerSubPix(img_thermal, corners3, (11, 11), (-1, -1), criteria)
        imgpoints_thermal_individual.append(corners_refined3)
        objpoints_thermal_individual.append(objp)

    # Sincronizar detecciones para calibración estéreo entre cámaras color
    if ret1 and ret2:
        objpoints_stereo_color.append(objp)
        imgpoints_color1_stereo.append(corners_refined1)
        imgpoints_color2_stereo.append(corners_refined2)

    # Sincronizar detecciones para calibración estéreo entre cámara térmica y color 1
    if ret1 and ret3:
        objpoints_stereo_thermal.append(objp)
        imgpoints_color1_thermal.append(corners_refined1)
        imgpoints_thermal.append(corners_refined3)

# Calibrar cámaras individualmente
_, mtx_color1, dist_color1, _, _ = cv2.calibrateCamera(objpoints_color1, imgpoints_color1, img_color1.shape[::-1], None, None)
_, mtx_color2, dist_color2, _, _ = cv2.calibrateCamera(objpoints_color2, imgpoints_color2, img_color2.shape[::-1], None, None)
_, mtx_thermal, dist_thermal, _, _ = cv2.calibrateCamera(objpoints_thermal_individual, imgpoints_thermal_individual, img_thermal.shape[::-1], None, None)

# Verificar que las listas para calibración estéreo tengan la misma longitud
print("Número de vistas para calibración estéreo color-color:", len(objpoints_stereo_color))
print("Número de vistas para calibración estéreo térmica-color1:", len(objpoints_stereo_thermal))

# Calibración estéreo entre cámaras color
flags = cv2.CALIB_FIX_INTRINSIC
ret_color, _, _, _, _, R_color, T_color, _, _ = cv2.stereoCalibrate(
    objpoints_stereo_color, imgpoints_color1_stereo, imgpoints_color2_stereo,
    mtx_color1, dist_color1, mtx_color2, dist_color2, img_color1.shape[::-1],
    flags=flags, criteria=criteria)

# Calibración estéreo entre cámara térmica y cámara color 1
ret_thermal, _, _, _, _, R_thermal, T_thermal, _, _ = cv2.stereoCalibrate(
    objpoints_stereo_thermal, imgpoints_color1_thermal, imgpoints_thermal,
    mtx_color1, dist_color1, mtx_thermal, dist_thermal, img_color1.shape[::-1],
    flags=flags, criteria=criteria)


# Rectificación de las cámaras
rect_color1, rect_color2, proj_color1, proj_color2, Q, _, _ = cv2.stereoRectify(
    mtx_color1, dist_color1, mtx_color2, dist_color2, img_color1.shape[::-1], R_color, T_color)

rect_thermal, rect_color1, proj_thermal, proj_color1, _, _, _ = cv2.stereoRectify(
    mtx_color1, dist_color1, mtx_thermal, dist_thermal, img_color1.shape[::-1], R_thermal, T_thermal)

# Mostrar resultados clave
print("Matriz intrínseca cámara color 1:", mtx_color1)
print("Matriz intrínseca cámara térmica:", mtx_thermal)
print("Matriz de rotación entre cámaras color:", R_color)
print("Matriz de traslación entre cámaras térmica y color 1:", T_thermal)



# Carga las imágenes para cada cámara
color1_images = glob.glob('../CalibrationData/Steven/left/*.png')
color2_images = glob.glob('../CalibrationData/Steven/right/*.png')
thermal_images = glob.glob('../CalibrationData/Steven/thermal/*.png')

# Asegúrate de que todas las listas de imágenes estén ordenadas y tengan la misma longitud
color1_images.sort()
color2_images.sort()
thermal_images.sort()


# Crear directorios para guardar las imágenes rectificadas si no existen
os.makedirs('../CalibrationData/Steven_rectified/left', exist_ok=True)
os.makedirs('../CalibrationData/Steven_rectified/right', exist_ok=True)
os.makedirs('../CalibrationData/Steven_rectified/thermal', exist_ok=True)

# Obtener los mapas de remapeo para las cámaras color
map1_x, map1_y = cv2.initUndistortRectifyMap(
    mtx_color1, dist_color1, rect_color1, proj_color1, img_color1.shape[::-1], cv2.CV_32FC1)
map2_x, map2_y = cv2.initUndistortRectifyMap(
    mtx_color2, dist_color2, rect_color2, proj_color2, img_color2.shape[::-1], cv2.CV_32FC1)

# Obtener los mapas de remapeo para la cámara térmica y color1
map_thermal_x, map_thermal_y = cv2.initUndistortRectifyMap(
    mtx_thermal, dist_thermal, rect_thermal, proj_thermal, img_thermal.shape[::-1], cv2.CV_32FC1)
map_color1_x_thermal, map_color1_y_thermal = cv2.initUndistortRectifyMap(
    mtx_color1, dist_color1, rect_color1, proj_color1, img_color1.shape[::-1], cv2.CV_32FC1)

# Procesar y rectificar las imágenes
for idx in range(num_images):
    # Leer imágenes originales
    img_color1 = cv2.imread(color1_images[idx])
    img_color2 = cv2.imread(color2_images[idx])
    img_thermal = cv2.imread(thermal_images[idx])

    # Rectificar imágenes de las cámaras color
    img_color1_rect = cv2.remap(img_color1, map1_x, map1_y, cv2.INTER_LINEAR)
    img_color2_rect = cv2.remap(img_color2, map2_x, map2_y, cv2.INTER_LINEAR)

    # Rectificar imágenes de la cámara térmica y color1
    img_thermal_rect = cv2.remap(img_thermal, map_thermal_x, map_thermal_y, cv2.INTER_LINEAR)
    img_color1_rect_thermal = cv2.remap(img_color1, map_color1_x_thermal, map_color1_y_thermal, cv2.INTER_LINEAR)

    # Guardar imágenes rectificadas
    cv2.imwrite(f'../CalibrationData/Steven_rectified/left/left_{idx}.png', img_color1_rect)
    cv2.imwrite(f'../CalibrationData/Steven_rectified/right/right_{idx}.png', img_color2_rect)
    cv2.imwrite(f'../CalibrationData/Steven_rectified/thermal/thermal_{idx}.png', img_thermal_rect)

    # Opcional: Visualizar las imágenes rectificadas lado a lado con líneas horizontales
    # Combinar imágenes para visualización
    height, width, _ = img_color1_rect.shape
    combined_color = np.hstack((img_color1_rect, img_color2_rect))
    combined_thermal = np.hstack((img_color1_rect_thermal, img_thermal_rect))

    # Dibujar líneas horizontales
    num_lines = 10
    step = height // num_lines
    for i in range(0, height, step):
        cv2.line(combined_color, (0, i), (2 * width, i), (0, 255, 0), 1)
        cv2.line(combined_thermal, (0, i), (2 * width, i), (0, 255, 0), 1)

    # Mostrar imágenes
    cv2.imshow('Rectified Color Cameras', combined_color)
    cv2.imshow('Rectified Thermal and Color1', combined_thermal)
    cv2.waitKey(0)  # Mostrar cada par por 500 ms

cv2.destroyAllWindows()
