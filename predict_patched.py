#import config
from re import L
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from PIL import Image, ImageOps
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from network_standard import UNet
import time


def make_predictions(state_dict_path, path):
	
	patch_size = 256
	stride = 32
	#device = 'cpu'
	
	
	# create the output path
	output_path = path+'_mask_xR/'
	boundary_path = path+'_boundary_XR/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	if not os.path.exists(boundary_path):
		os.makedirs(boundary_path)
		 
	# Load the state dictionary
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	state_dict = torch.load(state_dict_path, map_location=torch.device(device))
	
	# Initialize the model
	model = U_Net(img_ch=1,output_ch=2)
	# model = UNet(in_channels=3,out_channels=3)

	# Load the state dictionary into the model
	model.load_state_dict(state_dict)

	# count coefficients
	# for name, param in model.named_parameters():
	# 	print(f'Layer: {name} | Number of parameters: {param.numel()}')
		
	#total_params = sum(p.numel() for p in model.parameters())

	# Move the model to the desired device
	model = model.to(device)

	# # print model
	# for name, param in model.named_parameters():
	# 	print(name, param)
		
	# set model to evaluation mode
	model.eval()
	
	
	# get images
	image_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
    
	loss_cells = 0
	loss_neiclis = 0
	loss_boundaries = 0
	ref_matchc = 0
	ref_matchn = 0
	ref_matchb = 0

	# turn off gradient tracking
	with torch.no_grad():
		for index in range(len(image_paths)):
			
			# Add precceding and following images
			# if (index == 0):
			# 	image_path_prec = image_paths[index]
			# 	image_path_foll = image_paths[index+1]
			# elif (index == len(image_paths)-1):
			# 	image_path_prec = image_paths[index-1]
			# 	image_path_foll = image_paths[index]
			# else:
			# 	image_path_prec = image_paths[index-1]
			# 	image_path_foll = image_paths[index+1]
			# prec_image = Image.open(image_path_prec).convert("L")
			# foll_image = Image.open(image_path_foll).convert("L")
				
		
			if image_paths[index].endswith('.tif'):
				
				# Loading cell channels
				# image_path= image_paths[index]
				# filename = image_path.split('/')[-1][:-len(".tif")]
				# dirname = image_path.split('/')[:-2]
				# dirname = '/'.join(dirname)
				# framename = filename.split('_')[:-2]
				# framename = '_'.join(framename)
				# sectionnum = filename.split('_')[-1]			
				# image_pathC2 =  dirname + "/insulin/" + framename + "_Insulin_" + sectionnum + ".tif"
				# GT_paths = dirname+'/train_GT_binary/'
				# GT_pathC1 = GT_paths+"/" + framename + "_CellMask_" + sectionnum + ".tif"
				# GT_pathC2 = GT_paths+"/" + framename + "_NucleiMask_" + sectionnum + ".tif"
				# GT_pathC3 = GT_paths+"/" + framename + "_Borders_" + sectionnum + ".tif"
				# imagec1 = Image.open(image_path).convert("L")
				# imagec2 = Image.open(image_pathC2).convert("L")
				# dummy_channel = Image.new("L", imagec2.size)
				# GTc1 = Image.open(GT_pathC1)
				# GTc2 = Image.open(GT_pathC2)
				# GTc3 = Image.open(GT_pathC3)
				# GTc1 = np.array(GTc1)
				# GTc2 = np.array(GTc2)
				# GTc3 = np.array(GTc3)
				# GTc1[GTc1 > 0] = 1
				# GTc2[GTc2 > 0] = 1
				# GTc3[GTc3 > 0] = 1
				# GTc1 = Image.fromarray(GTc1).convert("L")
				# GTc2 = Image.fromarray(GTc2).convert("L")
				# GTc3 = Image.fromarray(GTc3).convert("L")
				# # merging the channels
				# image = Image.merge("RGB", [imagec1, imagec2, dummy_channel])
				# GT = Image.merge("RGB", [GTc1, GTc2, GTc3])
				# #GT_tensor = T.ToTensor()(GT).unsqueeze(0).to(device)
				# img_tensor = torch.from_numpy(np.array(image)).unsqueeze(0).float()
				# img_tensor = img_tensor.permute(0, 3, 1, 2)
				# GT_tensor = torch.from_numpy(np.array(GT)).unsqueeze(0).float()
				# GT_tensor = GT_tensor.permute(0, 3, 1, 2).to(device)




				image = Image.open(image_paths[index]).convert("L")
				# image_rotated = image.rotate(90, Image.NEAREST, expand = 1)
				
				# convert uint 16 to uint8 if needed
				# image_data = np.array(image)
				# image_uint8 = image_data.astype(np.uint8)
				# image = Image.fromarray(image_uint8)
				# image = image.convert('L')
				
				# predicate whole image at ones
				img_tensor = T.ToTensor()(image).unsqueeze(0)  # Convert to tensor and add batch dimension
				normalizer = nn.LayerNorm([1, 958, 303])			# xz section prediction
				#normalizer = nn.LayerNorm([1, 958, 1405])			# xy section prediction
				
				# zy section prediction
				# img_tensor = T.functional.rotate(img_tensor, 90)
				# normalizer = nn.LayerNorm([1, 303, 958])	
				 
				 
				# Standard UNet
				# normalizer = nn.LayerNorm([3, 1024, 1024])		
				

				# add the preceeding and following images as channels
				# image = Image.merge("RGB", (prec_image, image, foll_image))# check if the image is 3 channel or not
				# # predicate whole image at ones
				# img_tensor = T.ToTensor()(image)
				# normalizer = nn.LayerNorm([3, 958, 1405])
				img_tensor = normalizer(img_tensor)
				# img_tensor = img_tensor.unsqueeze(0) 
				

				# predMask, checkbtlnek , _, _, _=  model(img_tensor.to(device))
				# output = torch.sigmoid(predMask)
				img_tensor = img_tensor.to(device)

				# #Calculate forward propagation time
				# startTime = time.time()
				# for i in range (100):
				# 	predMask, checkbtlnek =  model(img_tensor)
				# end_time = time.time()
				# print(f'Elapsed time: {(end_time - startTime)/100}')
				# output = torch.sigmoid(predMask)
				


				# SR_cell = output[:, 0, :, :]
				# SR_neicli = output[:, 1, :, :]
				# SR_boundary = output[:, 2, :, :]
				# GT_cell = GT_tensor[:, 0, :, :]
				# GT_neicli = GT_tensor[:, 1, :, :]
				# GT_boundary = GT_tensor[:, 2, :, :]
				
				# loss_cell = diceLoss(SR_cell.view(SR_cell.size(0), -1), GT_cell.view(GT_cell.size(0), -1))
				# loss_neicli = diceLoss(SR_neicli.view(SR_neicli.size(0), -1), GT_neicli.view(GT_neicli.size(0), -1))
				# loss_boundary = diceLoss(SR_boundary.view(SR_boundary.size(0), -1), GT_boundary.view(GT_boundary.size(0), -1))
				
				# loss_cells += loss_cell
				# loss_neiclis += loss_neicli
				# loss_boundaries += loss_boundary

				# ref_cell = diceLoss(GT_cell.view(GT_cell.size(0), -1), GT_cell.view(GT_cell.size(0), -1))
				# ref_neicli = diceLoss(GT_neicli.view(GT_neicli.size(0), -1), GT_neicli.view(GT_neicli.size(0), -1))
				# ref_boundary = diceLoss(GT_boundary.view(GT_boundary.size(0), -1), GT_boundary.view(GT_boundary.size(0), -1))
				
				# ref_matchc += ref_cell
				# ref_matchn += ref_neicli
				# ref_matchb += ref_boundary

				# output = output.permute(0, 2, 3, 1)


				SR, _, _, _, _ = model(img_tensor)
				output = torch.sigmoid(SR)
				mask = output[:, 0, :, :]
				boundary = output[:, 1, :, :]
				mask = mask.squeeze(0)
				boundary = boundary.squeeze(0)
				mask = mask - mask.min()
				mask = mask / mask.max()
				mask[mask >= 0.5] = 255
				mask = mask.byte()
				mask_np = mask.cpu().numpy()
				mask_image = Image.fromarray(mask_np)
				mask_image.save(os.path.join(output_path, image_paths[index].split('/')[-1]))
				boundary = boundary - boundary.min()
				boundary = boundary / boundary.max()
				boundary[boundary >= 0.5] = 255
				boundary = boundary.byte()
				boundary_np = boundary.cpu().numpy()
				boundary_image = Image.fromarray(boundary_np)
				boundary_image.save(os.path.join(boundary_path, image_paths[index].split('/')[-1][:-len(".tif")] + "_boundary.tif"))

				# rotate back to original orientation
				#output = T.functional.rotate(output, -90)
				
				# normalize and save
				# output = output.squeeze(0)
				#output = output.squeeze(0)
				# output = output - output.min()
				# output = output / output.max()
				# #output[output >= 0.5] = 255
				# output = ((output) * 255) 
				# #output[output < 165] = 0

				# output = output.byte()
				# output_np = output.cpu().numpy()
				# output_image = Image.fromarray(output_np)
				
				# #output_image_rerotated = output_image.rotate(-90, Image.NEAREST, expand = 1)
				# output_image.save(os.path.join(output_path, image_paths[index].split('/')[-1]))


	# 			o1 = output[:,:,0]
	# 			o2 = output[:,:,1]
	# 			o3 = output[:,:,2]
	# 			o1 = o1- o1.min()
	# 			o1 = o1 / o1.max()
	# 			o2 = o2- o2.min()
	# 			o2 = o2 / o2.max()
	# 			o3 = o3- o3.min()
	# 			o3 = o3 / o3.max()
	# 			o1[o1>=0.5] = 255
	# 			o2[o2>=0.5] = 255
	# 			o3[o3>=0.5] = 255
	# 			o1 = o1.byte()
	# 			o2 = o2.byte()
	# 			o3 = o3.byte()
	# 			o1_np = o1.cpu().numpy()
	# 			o2_np = o2.cpu().numpy()
	# 			o3_np = o3.cpu().numpy()
	# 			o1_image = Image.fromarray(o1_np)
	# 			o2_image = Image.fromarray(o2_np)
	# 			o3_image = Image.fromarray(o3_np)
	# 			o1_image.save(os.path.join(output_path, image_paths[index].split('/')[-1][:-len(".tif")] + "_CellMask.tif"))
	# 			o2_image.save(os.path.join(output_path, image_paths[index].split('/')[-1][:-len(".tif")] + "_NucleiMask.tif"))
	# 			o3_image.save(os.path.join(output_path, image_paths[index].split('/')[-1][:-len(".tif")] + "_Borders.tif"))

	# print (f'loss_cell: {loss_cells}, loss_neicli: {loss_neiclis}, loss_boundary: {loss_boundaries}')
	# print (f'ref_cell: {ref_matchc}, ref_neicli: {ref_matchn}, ref_boundary: {ref_matchb}')
	# print (f'match percentage cell: {100* loss_cells/ref_matchc}, match percentage neicli: {100* loss_neiclis/ref_matchn}, match percentage boundery: {100* loss_boundaries/ref_matchb}' )


def diceLoss(x, y):
		intersection = torch.sum(x * y) + 1e-7
		union_squared = torch.sum(x**2) + torch.sum(y**2) + 1e-7
		return  2 * intersection / union_squared
		
		



