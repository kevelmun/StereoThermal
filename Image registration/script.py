from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
import torch
import matplotlib.pyplot as plt
import cv2
import glob

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
from torchvision.utils import save_image
import os

from scipy.stats import entropy
from scipy import ndimage
from skimage.metrics import normalized_mutual_information

from fusion import TIF
from metrics import metricsMutinf
from metrics import metricsPsnr
from metrics import metricsEdge_intensity
from metrics import metricsRmse
from metrics import metricsSsim
from metrics import metricsQcb
from metrics import metricsQcv

from skimage.metrics import structural_similarity as ssim
from skimage import filters
import time

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

print("device: ", device)

extractor_str_list=["superpoint", "disk", "sift", "aliked"]
#extractor = "superpoint" #"superpoint", "disk", "aliked", "sift", "doghardnet"

for extractor_str in extractor_str_list:
    if extractor_str=="superpoint":
        #extractor = SuperPoint(max_num_keypoints=1000).eval().to(device) # load the extractor
        #matcher = LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.9).eval().to(device) # load the matcher

        # SuperPoint+LightGlue
        #matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device) # load the matcher
        extractor = SuperPoint(max_num_keypoints=2048*2).eval().to(device) # load the extractor
        matcher = LightGlue(features='superpoint').eval().to(device) # load the matcher
    elif extractor_str=="disk":
        ## or DISK+LightGlue
        #matcher = LightGlue(features='disk', depth_confidence=0.9, width_confidence=0.95).eval().to(device) # load the matcher
        extractor = DISK(max_num_keypoints=2048*2).eval().to(device) # load the extractor
        matcher = LightGlue(features='disk').eval().to(device) # load the matcher
    elif extractor_str=="sift":
        ## or SIFT+LightGlue
        extractor = SIFT(max_num_keypoints=2048*2).eval().to(device) # load the extractor
        matcher = LightGlue(features='sift').eval().to(device) # load the matcher
    elif extractor_str=="aliked":
        ## or ALIKED+LightGlue
        extractor = ALIKED(max_num_keypoints=2048*2).eval().to(device) # load the extractor
        matcher = LightGlue(features='aliked').eval().to(device) # load the matcher
    elif extractor_str=="doghardnet":
        ## or DoGHardNet+LightGlue
        extractor = DoGHardNet(max_num_keypoints=2048*2).eval().to(device) # load the extractor
        matcher = LightGlue(features='doghardnet').eval().to(device) # load the matcher    

    reg_images="visible" #visible, thermal     
    cal_metrics=True
    dataset = "01"
    root = '../../Datasets/Dataset02Rafael_reg'
    #root = '../../Datasets/FLIR_TEST' 
    #root = 'C:/Respaldo/Henry/Datasets/M3FD_Fusion' 

    root_1 = root+'/'+dataset
    type_data="visible"
    suffix_thermal="th"
    suffix_visible="vis"
    extension_vis = ".bmp"
    extension_th  = ".bmp"
    extension_reg = ".bmp"
    method="reg_lightglue_"+extractor_str+"_vis"

    os.makedirs(root_1+"/"+method+"/", exist_ok=True)
    os.makedirs(root_1+"/"+method+"/combined/", exist_ok=True)
    #os.makedirs(root_1+"/reg_lightglue_ad/", exist_ok=True)
    #os.makedirs(root_1+"/tmp/", exist_ok=True)

    archi1=open(root_1+"/"+method+"/time.csv","w") 
    archi1.write("image;time\n") 

    root_ = root_1+"/"+type_data+"/"
    list_image_path = []
    list_image_path.extend(glob.glob(root_+'/*'+extension_vis))

    for img_path in list_image_path:    
        inicio = time.time()

        name_image=img_path.replace(root_1+"/"+type_data+"\\","")
           
        if type_data=="thermal":
            name_image=name_image.replace(suffix_thermal,"")
        elif type_data=="visible":
            name_image=name_image.replace(suffix_visible,"")
        name_image=name_image.replace(extension_vis,"")
        print("name_image: ",name_image)
        
        try:
            if reg_images=="thermal":
                image0 = load_image(root_1+'/thermal/'+name_image+suffix_thermal+extension_th).to(device)   # Assuming 'th_006.bmp' is the correct path to the image
                image1 = load_image(root_1+'/visible/'+name_image+suffix_visible+extension_vis).to(device) #.cuda()
            else: 
                image0 = load_image(root_1+'/visible/'+name_image+suffix_visible+extension_vis).to(device) #.cuda()
                image1 = load_image(root_1+'/thermal/'+name_image+suffix_thermal+extension_th).to(device)   # Assuming 'th_006.bmp' is the correct path to the image
                
            feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
            feats1 = extractor.extract(image1)
            
            ##pts0_ = feats0['keypoints'][0].cpu().numpy()
            ##pts1_ = feats1['keypoints'][0].cpu().numpy()
            
            ##print("feats0: ",type(pts0_))
            ##print("feats0: ",len(pts0_))
            ##print("feats1: ",len(pts1_))

            #M0, _ = cv2.findHomography(pts0_, pts1_)

            matches01 = matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
            
            #matches = matches01['matches']  # indices with shape (K,2)
            matches, scores = matches01["matches"], matches01["scores"]
            points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
            points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

            pts0 = points0.cpu().numpy()
            pts1 = points1.cpu().numpy()

            #print("pts0: ",matches[..., 0])
            #print("pts0: ",len(pts0))
            #print("pts1: ",len(pts1))


            # Find the homography matrix that maps all points from pts0 to pts1
            M1, _ = cv2.findHomography(pts0, pts1)
            
            #if cal_metrics:
                #print("Homography_0: ",M0)
                #print("Homography_1: ",M1)
                #print("scores: ", scores)

            #score = np.mean(M0 == M1)
            #print("Score: ",score) 
                
            # Convert the PyTorch image tensors to PIL images for warpPerspective
            image0_pil = to_pil_image(image0)
            image1_pil = to_pil_image(image1)

            # Apply perspective transformation using warpPerspective
            image0_warped = cv2.warpPerspective(np.array(image0_pil), M1, (image1.shape[2], image1.shape[1]))
            warped_image0 = torch.tensor(image0_warped).permute(2, 0, 1)

            warped_image0_pil = to_pil_image(warped_image0)

            # Convert the PIL image back to a PyTorch tensor
            warped_image0_tensor = torch.tensor(np.array(warped_image0_pil)).permute(2, 0, 1)

            # Normalize the tensor to the range [0, 1]
            warped_image0_tensor = warped_image0_tensor.float() / 255.0
            warped_image0_tensor = resize(warped_image0_tensor, (480,640))

            # Save the warped image to the "images" folder with a desired name
            if reg_images=="thermal":
                save_image(warped_image0_tensor, root_1+"/"+method+"/"+name_image+suffix_thermal+extension_reg)
            else:
                save_image(warped_image0_tensor, root_1+"/"+method+"/"+name_image+suffix_visible+extension_reg)
            
            if cal_metrics:
                fixed_image_np = np.transpose(image0.cpu().detach().numpy(), (1, 2, 0))
                moving_image_np = np.transpose(image1.cpu().detach().numpy(), (1, 2, 0))
                result_image_np = np.transpose(warped_image0.cpu().detach().numpy(), (1, 2, 0)) 
                
                fixed_image_gray = cv2.cvtColor(fixed_image_np, cv2.COLOR_BGR2GRAY)  
                result_image_gray = cv2.cvtColor(result_image_np, cv2.COLOR_BGR2GRAY)            
                
                """
                fig, ax = plt.subplots(1, 3, figsize=(18, 6))

                # Dibujar las imágenes originales
                #print("fixed_image_np: ",fixed_image_np.shape)
                ax[0].imshow(fixed_image_np)
                ax[0].set_title('Fixed Image')
                ax[0].axis('off')

                #print("moving_image_np: ",moving_image_np.shape)
                ax[1].imshow(moving_image_np)
                ax[1].set_title('Moving Image')
                ax[1].axis('off')

                # Dibujar los puntos coincidentes y las líneas
                ax[0].scatter(points0.cpu().detach().numpy()[:, 0], points0.cpu().detach().numpy()[:, 1], c='b', s=5)             
                ax[1].scatter(points1.cpu().detach().numpy()[:, 0], points1.cpu().detach().numpy()[:, 1], c='b', s=5)

                # Mostrar la imagen0 transformada
                #print("result_image_np: ",result_image_np.shape)
                ax[2].imshow(result_image_np)
                ax[2].set_title('Result Image')
                ax[2].axis('off')
                #plt.show()               
                """

                # Crear una nueva imagen combinando las dos imágenes originales
                combined_image = np.hstack((fixed_image_np, moving_image_np))

                # Crear una nueva figura y un solo eje para mostrar la imagen combinada
                plt.figure(figsize=(12, 6))

                # Mostrar la imagen combinada
                plt.imshow(combined_image)
                #plt.title('Correspondence of Points')
                plt.axis('off')
                
                #for i in range(len(points0)):
                #    color = plt.cm.viridis(scores[i])  # Color based on score
                #    plt.plot([points0[i, 0], points1[i, 0] + fixed_image_np.shape[1]], [points0[i, 1], points1[i, 1]], c=color, linewidth=0.5)

                # Dibujar líneas entre los puntos coincidentes en la imagen combinada
                for i in range(len(points0)):
                    score_tmp = scores[i].item()
                    if score_tmp >= 0.90:
                        plt.plot([points0[i, 0], points1[i, 0] + fixed_image_np.shape[1]], [points0[i, 1], points1[i, 1]], c='g', linewidth=0.5)
                    elif score_tmp >= 0.75 and score_tmp < 0.90:
                        plt.plot([points0[i, 0], points1[i, 0] + fixed_image_np.shape[1]], [points0[i, 1], points1[i, 1]], c='#FFFF00', linewidth=0.5)
                    elif score_tmp >= 0.60 and score_tmp < 0.75:
                        plt.plot([points0[i, 0], points1[i, 0] + fixed_image_np.shape[1]], [points0[i, 1], points1[i, 1]], c='#FFA500', linewidth=0.5)
                    elif score_tmp >= 0.50 and score_tmp < 0.60:
                        plt.plot([points0[i, 0], points1[i, 0] + fixed_image_np.shape[1]], [points0[i, 1], points1[i, 1]], c='r', linewidth=0.5)
                        
                    if score_tmp >= 0.50:
                        # Plotear los puntos en la imagen combinada
                        plt.scatter(points0[i, 0], points0[i, 1], c='b', label='Points 0', s=5)
                        plt.scatter(points1[i, 0] + fixed_image_np.shape[1], points1[i, 1], c='b', label='Points 1', s=5)
                
                # Guardar la imagen combinada con los puntos y las líneas dibujadas
                plt.savefig(root_1+'/'+method+'/combined/'+name_image+'_combined.png', bbox_inches='tight', pad_inches=0)
                
                #plt.show()
                plt.close()
                
        except Exception as err:
            #archi1.write(name_image+";"+str(0.0)+";"+str(0.0)+";"+str(0.0)+";"+str(0.0) + "\n") 
            print(f"Unexpected {err=}, {type(err)=}")
            
        fin = time.time()
        print("Elapsed time: "+str(fin-inicio))
        archi1.write(name_image+";"+str(fin-inicio)+ "\n") 

    archi1.close()






"""
import torch
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
from torchvision.utils import save_image
import os
from scipy.stats import entropy
from scipy import ndimage
from skimage.metrics import normalized_mutual_information
from fusion import TIF
from metrics import metricsMutinf, metricsPsnr, metricsEdge_intensity, metricsRmse, metricsSsim, metricsQcb, metricsQcv
from skimage.metrics import structural_similarity as ssim
from skimage import filters
import time
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device: ", device)

extractor_str_list = ["aliked"]

for extractor_str in extractor_str_list:
    if extractor_str == "superpoint":
        extractor = SuperPoint(max_num_keypoints=2048*2).eval().to(device)
        matcher = LightGlue(features='superpoint').eval().to(device)
    elif extractor_str == "disk":
        extractor = DISK(max_num_keypoints=2048*2).eval().to(device)
        matcher = LightGlue(features='disk').eval().to(device)
    elif extractor_str == "sift":
        extractor = SIFT(max_num_keypoints=2048*2).eval().to(device)
        matcher = LightGlue(features='sift').eval().to(device)
    elif extractor_str == "aliked":
        extractor = ALIKED(max_num_keypoints=2048*2).eval().to(device)
        matcher = LightGlue(features='aliked').eval().to(device)
    elif extractor_str == "doghardnet":
        extractor = DoGHardNet(max_num_keypoints=2048*2).eval().to(device)
        matcher = LightGlue(features='doghardnet').eval().to(device)

    reg_images = "visible"
    cal_metrics = True
    dataset = "01"
    root = '../../Datasets/Dataset02Rafael_reg'
    root_1 = root + '/' + dataset
    type_data = "visible"
    suffix_thermal = "th"
    suffix_visible = "vis"
    extension_vis = ".bmp"
    extension_th = ".bmp"
    extension_reg = ".bmp"
    method = "reg_lightglue_" + extractor_str + "_vis"

    os.makedirs(root_1 + "/" + method + "/", exist_ok=True)
    os.makedirs(root_1 + "/" + method + "/combined/", exist_ok=True)

    archi1 = open(root_1 + "/" + method + "/time.csv", "w")
    archi1.write("image;time\n")

    root_ = root_1 + "/" + type_data + "/"
    list_image_path = []
    list_image_path.extend(glob.glob(root_ + '/*' + extension_vis))

    score_threshold = 0.5  # Umbral de filtrado de matches

    for img_path in list_image_path:
        inicio = time.time()
        name_image = img_path.replace(root_1 + "/" + type_data + "\\", "")
        if type_data == "thermal":
            name_image = name_image.replace(suffix_thermal, "")
        elif type_data == "visible":
            name_image = name_image.replace(suffix_visible, "")
        name_image = name_image.replace(extension_vis, "")
        print("name_image: ", name_image)

        try:
            if reg_images == "thermal":
                image0 = load_image(root_1 + '/thermal/' + name_image + suffix_thermal + extension_th).to(device)
                image1 = load_image(root_1 + '/visible/' + name_image + suffix_visible + extension_vis).to(device)
            else:
                image0 = load_image(root_1 + '/visible/' + name_image + suffix_visible + extension_vis).to(device)
                image1 = load_image(root_1 + '/thermal/' + name_image + suffix_thermal + extension_th).to(device)

            feats0 = extractor.extract(image0)
            feats1 = extractor.extract(image1)

            matches01 = matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            matches, scores = matches01["matches"], matches01["scores"]
            points0 = feats0['keypoints'][matches[..., 0]]
            points1 = feats1['keypoints'][matches[..., 1]]

            # Filtrar matches según el umbral de score
            mask = scores >= score_threshold
            points0 = points0[mask]
            points1 = points1[mask]
            scores = scores[mask]

            pts0 = points0.cpu().numpy()
            pts1 = points1.cpu().numpy()

            M1, _ = cv2.findHomography(pts0, pts1)

            image0_pil = to_pil_image(image0)
            image1_pil = to_pil_image(image1)

            image0_warped = cv2.warpPerspective(np.array(image0_pil), M1, (image1.shape[2], image1.shape[1]))
            warped_image0 = torch.tensor(image0_warped).permute(2, 0, 1)

            warped_image0_pil = to_pil_image(warped_image0)
            warped_image0_tensor = torch.tensor(np.array(warped_image0_pil)).permute(2, 0, 1)
            warped_image0_tensor = warped_image0_tensor.float() / 255.0
            warped_image0_tensor = resize(warped_image0_tensor, (480, 640))

            if reg_images == "thermal":
                save_image(warped_image0_tensor, root_1 + "/" + method + "/" + name_image + suffix_thermal + extension_reg)
            else:
                save_image(warped_image0_tensor, root_1 + "/" + method + "/" + name_image + suffix_visible + extension_reg)

            if cal_metrics:
                fixed_image_np = np.transpose(image0.cpu().detach().numpy(), (1, 2, 0))
                moving_image_np = np.transpose(image1.cpu().detach().numpy(), (1, 2, 0))
                result_image_np = np.transpose(warped_image0.cpu().detach().numpy(), (1, 2, 0))

                fig, ax = plt.subplots(1, 3, figsize=(18, 6))

                ax[0].imshow(fixed_image_np)
                ax[0].set_title('Fixed Image')
                ax[0].axis('off')

                ax[1].imshow(moving_image_np)
                ax[1].set_title('Moving Image')
                ax[1].axis('off')

                ax[0].scatter(points0.cpu().detach().numpy()[:, 0], points0.cpu().detach().numpy()[:, 1], c='r', s=5)
                ax[1].scatter(points1.cpu().detach().numpy()[:, 0], points1.cpu().detach().numpy()[:, 1], c='r', s=5)

                ax[2].imshow(result_image_np)
                ax[2].set_title('Result Image')
                ax[2].axis('off')

                combined_image = np.hstack((fixed_image_np, moving_image_np))
                plt.figure(figsize=(12, 6))
                plt.imshow(combined_image)
                plt.axis('off')

                for i in range(len(points0)):
                    color = plt.cm.viridis(scores[i])  # Color based on score
                    plt.plot([points0[i, 0], points1[i, 0] + fixed_image_np.shape[1]], [points0[i, 1], points1[i, 1]], c=color, linewidth=0.5)

                plt.scatter(points0[:, 0], points0[:, 1], c='r', label='Points 0', s=5)
                plt.scatter(points1[:, 0] + fixed_image_np.shape[1], points1[:, 1], c='r', label='Points 1', s=5)

                plt.savefig(root_1 + '/' + method + '/combined/' + name_image + '_combined.png', bbox_inches='tight', pad_inches=0)
                plt.close()

        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")

        fin = time.time()
        print("Elapsed time: " + str(fin - inicio))
        archi1.write(name_image + ";" + str(fin - inicio) + "\n")

    archi1.close()
    
"""