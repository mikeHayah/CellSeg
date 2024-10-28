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
	
	criterion = torch.nn.BCELoss()
	# create the output path
	output_path = path+'_mask_xR/'
	neuli_path = path+'_nucli_xR/'
	boundary_path = path+'_boundary_XR/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	if not os.path.exists(boundary_path):
		os.makedirs(boundary_path)
	if not os.path.exists(neuli_path):
		os.makedirs(neuli_path)
		 
	# Load the state dictionary
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	state_dict = torch.load(state_dict_path, map_location=torch.device(device))
	
	# Initialize the model
	model = UNet(in_channels=2,out_channels=3)

	# Load the state dictionary into the model
	model.load_state_dict(state_dict)


	# Move the model to the desired device
	model = model.to(device)

		
	# set model to evaluation mode
	model.eval()
	model = model.cuda()
	
	
	# get images
	image_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
    
	loss_cells = 0
	loss_neiclis = 0
	loss_boundaries = 0
	

	# turn off gradient tracking
	with torch.no_grad():
		for index in range(len(image_paths)):				
		
			if image_paths[index].endswith('.tif'):
				
				# Loading cell channels
				image_path= image_paths[index]
				filename = image_path.split('/')[-1][:-len(".tif")]
				dirname = image_path.split('/')[:-2]
				dirname = '/'.join(dirname)
				framename = filename.split('_')[:-2]
				framename = '_'.join(framename)
				sectionnum = filename.split('_')[-1]			
				image_pathC2 =  dirname + "/insulin/" + framename + "_Insulin_" + sectionnum + ".tif"
				GT_paths = dirname+'/train_GT_binary/'
				GT_pathC1 = GT_paths+"/" + framename + "_CellMask_" + sectionnum + ".tif"
				GT_pathC2 = GT_paths+"/" + framename + "_NucleiMask_" + sectionnum + ".tif"
				GT_pathC3 = GT_paths+"/" + framename + "_Borders_" + sectionnum + ".tif"
				imagec1 = Image.open(image_path).convert("L")
				imagec1_tensor = T.ToTensor()(imagec1).unsqueeze(0)
				imagec2 = Image.open(image_pathC2).convert("L")
				imagec2_tensor = T.ToTensor()(imagec2).unsqueeze(0)
				img_tensor = torch.cat((imagec1_tensor,imagec2_tensor),dim=1)
				
				# dummy_channel = Image.new("L", imagec2.size)
				GTc1 = Image.open(GT_pathC1)
				GTc2 = Image.open(GT_pathC2)
				GTc3 = Image.open(GT_pathC3)
				GTc1 = np.array(GTc1)
				GTc2 = np.array(GTc2)
				GTc3 = np.array(GTc3)
				GTc1[GTc1 > 0] = 1
				GTc2[GTc2 > 0] = 1
				GTc3[GTc3 > 0] = 1
				GTc1 = Image.fromarray(GTc1).convert("L")
				GTc2 = Image.fromarray(GTc2).convert("L")
				GTc3 = Image.fromarray(GTc3).convert("L")
				# # merging the channels
				GT = Image.merge("RGB", [GTc1, GTc2, GTc3])
				GT_tensor = torch.from_numpy(np.array(GT)).unsqueeze(0).float()
				GT_tensor = GT_tensor.permute(0, 3, 1, 2).to(device)

				# Standard UNet
				normalizer = nn.LayerNorm([2, 1024, 1024])		
			
				img_tensor = normalizer(img_tensor)
				
				img_tensor = img_tensor.cuda()
				SR, _= model(img_tensor)
				output = torch.sigmoid(SR)
				mask = output[:, 0, :, :]
				neicli = output[:, 1, :, :]
				boundary = output[:, 2, :, :]


				GT_cell = GT_tensor[:, 0, :, :]
				GT_neicli = GT_tensor[:, 1, :, :]
				GT_boundary = GT_tensor[:, 2, :, :]
				
				loss_cell = diceLoss(mask.view(mask.size(0), -1), GT_cell.view(GT_cell.size(0), -1))
				loss_neicli = diceLoss(neicli.view(neicli.size(0), -1), GT_neicli.view(GT_neicli.size(0), -1))
				loss_boundary = diceLoss(boundary.view(boundary.size(0), -1), GT_boundary.view(GT_boundary.size(0), -1))		
				
				loss_cells += loss_cell
				loss_neiclis += loss_neicli
				loss_boundaries += loss_boundary

				# save the output
				mask = mask.squeeze(0)
				neicli = neicli.squeeze(0)
				boundary = boundary.squeeze(0)
				mask = mask - mask.min()
				mask = mask / mask.max()
				mask[mask >= 0.5] = 255
				mask = mask.byte()
				mask_np = mask.cpu().numpy()
				mask_image = Image.fromarray(mask_np)
				mask_image.save(os.path.join(output_path, image_paths[index].split('/')[-1]))
				neicli = neicli - neicli.min()
				neicli = neicli / neicli.max()
				neicli[neicli >= 0.5] = 255
				neicli = neicli.byte()
				neicli_np = neicli.cpu().numpy()
				neicli_image = Image.fromarray(neicli_np)
				neicli_image.save(os.path.join(neuli_path, image_paths[index].split('/')[-1][:-len(".tif")] + "_neicli.tif"))
				boundary = boundary - boundary.min()
				boundary = boundary / boundary.max()
				boundary[boundary >= 0.5] = 255
				boundary = boundary.byte()
				boundary_np = boundary.cpu().numpy()
				boundary_image = Image.fromarray(boundary_np)
				boundary_image.save(os.path.join(boundary_path, image_paths[index].split('/')[-1][:-len(".tif")] + "_boundary.tif"))
				


	print (f'loss_cell: {loss_cells}, loss_neicli: {loss_neiclis}, loss_boundary: {loss_boundaries}')
	print (f'avg loss_cell: {loss_cells/len(image_paths)}, avg loss_neicli: {loss_neiclis/len(image_paths)}, avg loss_boundary: {loss_boundaries/len(image_paths)}')
	

def diceLoss(x, y):
		intersection = torch.sum(x * y) + 1e-7
		union = torch.sum(x) + torch.sum(y) + 1e-7
		return  2 * intersection / union
		
		



