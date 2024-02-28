import torch
import glob
import numpy as np
#from ignite.metrics import PSNR
from tqdm import tqdm
import cv2
import math
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse
import argparse
import os
import torchmetrics

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--images_path1", type=str, default="expert_a_testing_set", help="input path of the GT set")
parser.add_argument("--images_path2", type=str, default="predicted_images", help="input path of the corrected images")
parser.add_argument("--metric", type=str, default="all", help="which metrich would you like to calculate? Options: [PSNR, L1, L2, SSIM]")
args = parser.parse_args()

"""
GT_A = sorted(glob.glob("../dataset/exposure/test/expert_a_testing_set/*"))
GT_B = sorted(glob.glob("../dataset/exposure/test/expert_b_testing_set/*"))
GT_C = sorted(glob.glob("../dataset/exposure/test/expert_c_testing_set/*"))
GT_D = sorted(glob.glob("../dataset/exposure/test/expert_d_testing_set/*"))
GT_E = sorted(glob.glob("../dataset/exposure/test/expert_e_testing_set/*"))
"""

GT = sorted(glob.glob(os.path.join(args.images_path1, "*")))
predicted_images = sorted(glob.glob(os.path.join(args.images_path2, "*")))

def calculate_PSNR(MSE, MAX_pixel = 255.0, torch_version=False):
	"""
		PSNR implementation
		For the pytorch version, please visit https://pytorch.org/ignite/generated/ignite.metrics.PSNR.html
		MSE: Calculated mean squared error
		MAX_pixel: The maximum possible pixel value
	"""
	if torch_version:
		return torchmetrics.functional.peak_signal_noise_ratio(image1, image2)
	else:
		return 20 * math.log10(MAX_pixel / math.sqrt(MSE))
	#return 10 * math.log10((MAX_pixel * MAX_pixel) / MSE)

def L1_distance(image1, image2):
	return np.average(np.abs(image1 - image2))

def L2_distance(image1, image2):
	#return np.linalg.norm(image1 - image2)
	return (mse(image1[:,:,0], image2[:,:,0]) + mse(image1[:,:,1], image2[:,:,1]) + mse(image1[:,:,2], image2[:,:,2])) / 3

def calculate_SSIM(image1, image2, data_range=2, multichannel=True, torch_version=False):
	"""
		data_range: the absolute difference between the maximum possible pixel value and the minimum possible pixel value.
		multichannel: whether the image has multichannel.
	"""
	if torch_version:
		return torchmetrics.StructuralSimilarityIndexMeasure(data_range=data_range)
	else:
		return ssim(image1, image2, data_range=data_range, multichannel=True)

PSNR = SSIM = L1 = L2 = 0.0

for i in tqdm(range(len(predicted_images))):
	img1 = cv2.imread(GT[i])
	img2 = cv2.imread(predicted_images[i])

	img1 = cv2.resize(img1, (256, 256))
	img2 = cv2.resize(img2, (256, 256))

	img1 = img1 / 255.0
	img2 = img2 / 255.0

	if args.metric == "PSNR" or args.metric == "all":
		MSE = L2_distance(img1, img2)
		PSNR += calculate_PSNR(MSE, MAX_pixel=1)
	if args.metric == "L1" or args.metric == "all":
		L1 += L1_distance(img1, img2)
	if args.metric == "L2" or args.metric == "all":
		L2 += L2_distance(img1, img2)
	if args.metric == "SSIM" or args.metric == "all":
		SSIM += calculate_SSIM(img1, img2, data_range=1, multichannel=True)

if args.metric == "PSNR" or args.metric == "all":
	PSNR = PSNR / len(predicted_images)
	print("PSNR: {}" .format(PSNR))
if args.metric == "L1" or args.metric == "all":
	L1 = L1 / len(predicted_images)
	print("L1: {}" .format(L1))
if args.metric == "L2" or args.metric == "all":
	L2 = L2 / len(predicted_images)
	print("L2: {}" .format(L2))
if args.metric == "SSIM" or args.metric == "all":
	SSIM = SSIM / len(predicted_images)
	print("SSIM: {}" .format(SSIM))
print("DONE!")





