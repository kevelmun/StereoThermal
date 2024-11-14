import sys
sys.path.append('core')
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from Disparity.Selective_IGEV.core.igev_stereo import IGEVStereo
from Disparity.Selective_IGEV.core.utils.utils import InputPadder
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F
import cv2

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_image_from_array(image_array):
    img = torch.from_numpy(image_array).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def get_SELECTIVE_disparity_map(
        img_left_array, img_right_array, restore_ckpt, 
        output_directory="selective_demo_output", save_numpy=False, mixed_precision=False, valid_iters=32, 
        hidden_dims=[128]*3, corr_implementation="reg", shared_backbone=False, corr_levels=2, 
        corr_radius=4, n_downsample=2, slow_fast_gru=False, n_gru_layers=3, max_disp=192):
    
    class Args:
        def __init__(self):
            self.restore_ckpt = restore_ckpt
            self.save_numpy = save_numpy
            self.output_directory = output_directory
            self.mixed_precision = mixed_precision
            self.valid_iters = valid_iters
            self.hidden_dims = hidden_dims
            self.corr_implementation = corr_implementation # options: ["reg", "alt", "reg_cuda", "alt_cuda"] if possible use reg_cuda for better performance
            self.shared_backbone = shared_backbone
            self.corr_levels = corr_levels
            self.corr_radius = corr_radius
            self.n_downsample = n_downsample
            self.slow_fast_gru = slow_fast_gru
            self.n_gru_layers = n_gru_layers
            self.max_disp = max_disp
    
    args = Args()

    left_images = [img_left_array]
    right_images = [img_right_array]

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    disparity_maps = []

    with torch.no_grad():
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (image1_array, image2_array) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image_from_array(image1_array)
            image2 = load_image_from_array(image2_array)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp).squeeze()
            disp = disp.cpu().numpy().squeeze()

            disparity_maps.append(disp)

            # Genera un nombre de archivo único o usa un índice para guardar las imágenes
            file_stem = f"output_{len(disparity_maps)}"
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp)
            plt.imsave(output_directory / f"{file_stem}.png", disp, cmap='jet')

    return disparity_maps

# Ejemplo de uso de la función con parámetros por defecto
if __name__ == "__main__":
    # Cargar las imágenes como arrays
    ""
    img_left = cv2.imread("../../images/calibration_results/matlab_1/flexometer/150/14_03_37_13_05_2024_IMG_LEFT.jpg")
    img_right = cv2.imread("../../images/calibration_results/matlab_1/flexometer/150/14_03_37_13_05_2024_IMG_RIGHT.jpg")
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)



    disparity_maps = get_SELECTIVE_disparity_map(
        img_left, img_right, 
        restore_ckpt="pretrained_models/middlebury_train.pth"
    )

    # Visualiza el primer mapa de disparidad generado
    plt.imshow(disparity_maps[0], cmap='jet')
    plt.show()
