import glob
import cv2
import os
import json
import numpy as np

def load_stereo_parameters(path):
    with open(path, 'r') as file:
        params = json.load(file)
        params['cameraMatrix1'] = np.transpose(np.array(params['cameraMatrix1']))
        params['cameraMatrix2'] = np.transpose(np.array(params['cameraMatrix2']))
        params['distCoeffs1'] = np.array(params['distCoeffs1'])
        params['distCoeffs2'] = np.array(params['distCoeffs2'])
        params['imageSize'] = tuple([params['imageSize'][1], params['imageSize'][0]])
        params['stereoR'] = np.transpose(np.array(params['stereoR']))
        params['stereoT'] = np.array(params['stereoT'])
    return params

def stereo_rectify(params):
    
    return cv2.stereoRectify(
        params['cameraMatrix1'], params['distCoeffs1'], params['cameraMatrix2'], params['distCoeffs2'], params['imageSize'],
        params['stereoR'], params['stereoT'], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

def create_rectify_map(cameraMatrix, distCoeffs, rectification, projection, imageSize):
    return cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, rectification,
                                       projection, imageSize, cv2.CV_16SC2)

def get_stereo_map_parameter(file_name, parameter):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    ret = fs.getNode(parameter).mat()
    fs.release()
    return ret

def get_stereo_map_parameters(file_name):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    stereoMapL_x = fs.getNode("stereoMapL_x").mat()
    stereoMapL_y = fs.getNode("stereoMapL_y").mat()
    stereoMapR_x = fs.getNode("stereoMapR_x").mat()
    stereoMapR_y = fs.getNode("stereoMapR_y").mat()

    fs.release()
    return stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y


def remap_image(image_path, map1, map2):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    rectified_image = cv2.remap(image, map1, map2, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    return rectified_image


if __name__ == "__main__":
    config_stereo = "../CalibrationConfig/ParamsStereo.json"
    config_thermal = "../CalibrationConfig/ParamsThermal.json"
    
    config_stereo_file = load_stereo_parameters(config_stereo)
    config_thermal_file = load_stereo_parameters(config_thermal)

    rect_color1, rect_color2, proj_color1, proj_color2, Q, _, _ = stereo_rectify(config_stereo_file)
    rect_color1, rect_thermal, proj_color1, proj_thermal, Q , _, _ = stereo_rectify(config_thermal_file)

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

    mtx_color1, dist_color1, img_color1 = config_stereo_file['cameraMatrix1'], config_stereo_file['distCoeffs1'], config_stereo_file['imageSize']

    # Obtener los mapas de remapeo para las cámaras color
    map1_x, map1_y = cv2.initUndistortRectifyMap(
        mtx_color1, dist_color1, rect_color1, proj_color1, img_color1.shape[::-1], cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(
        mtx_color2, dist_color2, rect_color2, proj_color2, img_color2.shape[::-1], cv2.CV_32FC1)

    # Obtener los mapas de remapeo para la cámara color1 y térmica
    map_color1_x_thermal, map_color1_y_thermal = cv2.initUndistortRectifyMap(
        mtx_color1, dist_color1, rect_color1, proj_color1, img_color1.shape[::-1], cv2.CV_32FC1)
    map_thermal_x, map_thermal_y = cv2.initUndistortRectifyMap(
        mtx_thermal, dist_thermal, rect_thermal, proj_thermal, img_thermal.shape[::-1], cv2.CV_32FC1)


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

