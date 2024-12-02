import cv2
from Cameras.util import process_images_in_directory
from util import *
from Image_registration import registration
import cv2
from util import *
from Image_registration import registration
from Fusion.fusion import *
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from testing_registration import merge_and_display_differences
import tifffile as tf

def main(imageFixed, imageMoved, threshold=200):
    
      # Procesar las imágenes utilizando el registro
    image_registered_bgr = registration.procesar_imagenes(ruta_imagen0=imageMoved, ruta_imagen1=imageFixed, threshold=threshold)
    # show_image(image_registered_bgr, 'Warped Image')


    fusioned_image_bgrt = fusion_bgr_lwir(imageFixed, image_registered_bgr)
    
    rgbt_image = extract_channels(fusioned_image_bgrt, [2,1,0,3])


    tf.imwrite(
        "rgbt.tiff", 
        rgbt_image, 
        photometric='rgb', 
        metadata={'description':'RGB Image + Thermal LWIR Channel'},
        compression=None)
    
    
    

    
    

if __name__ == "__main__":
    
    # Rutas de las imágenes
    image_1_path = "Cameras/captures/video_image_extractor_results/thermal/image_00005.png"
    image_0_path = "Cameras/captures/video_image_extractor_results/left/image_00005.png"
    
    # Umbral para el registro
    threshold = 250
    

    # main(imageFixed=image_0_path, imageMoved=image_1_path, threshold=200)

    

    print("END")






