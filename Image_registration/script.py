from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
import torch
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
import os
import time
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'
print("Device:", device)

extractor_str_list = ["superpoint", "disk", "sift", "aliked"]

for extractor_str in extractor_str_list:
    if extractor_str == "superpoint":
        extractor = SuperPoint(max_num_keypoints=2048 * 2).eval().to(device)
        matcher = LightGlue(features='superpoint').eval().to(device)
    elif extractor_str == "disk":
        extractor = DISK(max_num_keypoints=2048 * 2).eval().to(device)
        matcher = LightGlue(features='disk').eval().to(device)
    elif extractor_str == "sift":
        extractor = SIFT(max_num_keypoints=2048 * 2).eval().to(device)
        matcher = LightGlue(features='sift').eval().to(device)
    elif extractor_str == "aliked":
        extractor = ALIKED(max_num_keypoints=2048 * 2).eval().to(device)
        matcher = LightGlue(features='aliked').eval().to(device)
    elif extractor_str == "doghardnet":
        extractor = DoGHardNet(max_num_keypoints=2048 * 2).eval().to(device)
        matcher = LightGlue(features='doghardnet').eval().to(device)

    reg_images = "visible"  # Opciones: 'visible', 'thermal'
    cal_metrics = True
    root = "../captures"

    type_data = "visible/left"
    suffix_thermal = "thermal_"
    suffix_visible = "LEFT_visible_"
    extension_vis = ".png"
    extension_th = ".png"
    extension_reg = ".png"
    method = "reg_lightglue_" + extractor_str + "_vis"

    os.makedirs(os.path.join(root, method), exist_ok=True)
    os.makedirs(os.path.join(root, method, "combined"), exist_ok=True)

    root_ = os.path.join(root, type_data)
    list_image_path = glob.glob(os.path.join(root_, '*' + extension_vis))

    for img_path in list_image_path:
        inicio = time.time()

        name_image = os.path.basename(img_path)
        name_image = name_image.replace(suffix_visible, "").replace(extension_vis, "")
        print("Processing image:", name_image)

        try:
            thermal_image_path = os.path.join(root, 'thermal', suffix_thermal + name_image + extension_th)
            if not os.path.exists(thermal_image_path):
                print("Thermal image not found:", thermal_image_path)
                continue

            # Cargar imágenes
            image0 = load_image(img_path).to(device)
            image1 = load_image(thermal_image_path).to(device)

            # Redimensionar imágenes al mismo tamaño
            desired_size = (480, 640)  # (height, width)
            image0 = F.interpolate(image0.unsqueeze(0), size=desired_size, mode='bilinear', align_corners=False).squeeze(0)
            image1 = F.interpolate(image1.unsqueeze(0), size=desired_size, mode='bilinear', align_corners=False).squeeze(0)

            # Extraer características sin redimensionar
            feats0 = extractor.extract(image0, resize=None)
            feats1 = extractor.extract(image1, resize=None)

            matches01 = matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            matches, scores = matches01["matches"], matches01["scores"]

            # Verificar si hay suficientes correspondencias
            if len(matches) < 4:
                print(f"Not enough matches ({len(matches)}) for image: {name_image}. Skipping...")
                continue

            points0 = feats0['keypoints'][matches[..., 0]]
            points1 = feats1['keypoints'][matches[..., 1]]

            pts0 = points0.cpu().numpy()
            pts1 = points1.cpu().numpy()

            # Encontrar la matriz de homografía
            M1, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
            if M1 is None:
                print(f"Homography could not be computed for image: {name_image}. Skipping...")
                continue

            # Convertir imágenes a formato PIL
            image0_pil = to_pil_image(image0.cpu())
            image1_pil = to_pil_image(image1.cpu())

            # Aplicar transformación de perspectiva
            image0_warped = cv2.warpPerspective(np.array(image0_pil), M1, (image1.shape[2], image1.shape[1]))
            warped_image0 = torch.tensor(image0_warped).permute(2, 0, 1).to(device)

            # Normalizar
            warped_image0_tensor = warped_image0.float() / 255.0

            # Guardar la imagen transformada
            if reg_images == "thermal":
                output_image_name = suffix_thermal + name_image + extension_reg
            else:
                output_image_name = suffix_visible + name_image + extension_reg

            output_image_path = os.path.join(root, method, output_image_name)
            save_image(warped_image0_tensor, output_image_path)

            if cal_metrics:
                # Convertir imágenes a NumPy asegurando que están en CPU
                fixed_image_np = np.transpose(image0.cpu().detach().numpy(), (1, 2, 0))
                moving_image_np = np.transpose(image1.cpu().detach().numpy(), (1, 2, 0))
                result_image_np = np.transpose(warped_image0.cpu().detach().numpy(), (1, 2, 0))

                # Asegurar que las imágenes estén en formato uint8 para la visualización
                fixed_image_np = (fixed_image_np * 255).astype(np.uint8)
                moving_image_np = (moving_image_np * 255).astype(np.uint8)

                # Crear una imagen combinada para visualización
                combined_image = np.hstack((fixed_image_np, moving_image_np))

                # Crear y guardar la figura dentro de un bloque try-finally para asegurar que siempre se cierre
                try:
                    plt.figure(figsize=(12, 6))
                    plt.imshow(combined_image)
                    plt.axis('off')

                    for i in range(len(points0)):
                        score_tmp = scores[i].item()
                        if score_tmp >= 0.90:
                            color = 'g'
                        elif 0.75 <= score_tmp < 0.90:
                            color = '#FFFF00'
                        elif 0.60 <= score_tmp < 0.75:
                            color = '#FFA500'
                        elif 0.50 <= score_tmp < 0.60:
                            color = 'r'
                        else:
                            continue  # Saltar matches de baja confianza

                        # Dibujar líneas entre los puntos coincidentes
                        plt.plot([points0[i, 0], points1[i, 0] + fixed_image_np.shape[1]],
                                 [points0[i, 1], points1[i, 1]], c=color, linewidth=0.5)

                    # Guardar la imagen combinada con las líneas de correspondencia
                    combined_output_path = os.path.join(root, method, 'combined', name_image + '_combined.png')
                    plt.savefig(combined_output_path, bbox_inches='tight', pad_inches=0)
                finally:
                    plt.close()  # Asegurar que la figura se cierra siempre

        except cv2.error as cv_err:
            print(f"OpenCV error for image {name_image}: {cv_err}, type: {type(cv_err)}")
        except Exception as err:
            print(f"Unexpected error for image {name_image}: {err}, type: {type(err)}")

        fin = time.time()
        print("Elapsed time:", fin - inicio)
