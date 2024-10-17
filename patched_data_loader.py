from calendar import c
from ctypes import c_wchar
import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from torchvision.datasets import ImageFolder
from sklearn.utils import shuffle



class PatchImageFolder(ImageFolder):
	def __init__(self, root, patch_size=64, stride=32, mode="train" , augmentation_prob=0.4):
		#super(PatchImageFolder, self).__init__(root)
		"""Initializes image paths and preprocessing module."""
		self.root = root	
		# GT : Ground Truth
		self.GT_paths = root[:-1]+'_GT_binary/'
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.patch_size = patch_size
		self.stride = stride
		#self.resize = resize
		print("image count in {} path :{}".format(mode,len(self.image_paths)))
		
	def _extract_patches(self, image, GT, bGT):
		# if self.resize:
		# 	image = image.resize(self.resize, Image.ANTIALIAS)
			
		# img_tensor = T.ToTensor()(image).unsqueeze(0).float()  # Convert to tensor and add batch dimension
		# mute normalization as my mask is binary, in case of non-binary mask, use T.ToTensor()
		# GT_tensor = torch.from_numpy(np.array(GT)).unsqueeze(0).unsqueeze(0).float()	# Convert to tensor and add batch dimension
		img_tensor = torch.from_numpy(np.array(image)).unsqueeze(0).float()
		# img_tensor = img_tensor.permute(0, 3, 1, 2)
		GT_tensor = torch.from_numpy(np.array(GT)).unsqueeze(0).float()
		# GT_tensor = GT_tensor.permute(0, 3, 1, 2)
		bGT_tensor = torch.from_numpy(np.array(bGT)).unsqueeze(0).float()
		# GT_tensor = T.ToTensor()(GT).unsqueeze(0).float()
        
        # Extract patches using unfold
		img_patches = img_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
		img_patches = img_patches.contiguous().view(-1, 1, self.patch_size, self.patch_size)  # Flatten patches      
		
		GT_patches = GT_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
		GT_patches = GT_patches.contiguous().view(-1, 1, self.patch_size, self.patch_size)

		bGT_patches = bGT_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
		bGT_patches = bGT_patches.contiguous().view(-1, 1, self.patch_size, self.patch_size)
	
		return img_patches, GT_patches , bGT_patches
		
	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		# Add precceding and following images
		# if (index == 0):
		# 	image_path_prec = self.image_paths[index]
		# 	image_path_foll = self.image_paths[index+1]
		# elif (index == len(self.image_paths)-1):
		# 	image_path_prec = self.image_paths[index-1]
		# 	image_path_foll = self.image_paths[index]
		# else:
		# 	image_path_prec = self.image_paths[index-1]
		# 	image_path_foll = self.image_paths[index+1]
		
		# prec_image = Image.open(image_path_prec).convert("L")
		# foll_image = Image.open(image_path_foll).convert("L")
		
		# Read the images
		image_path = self.image_paths[index]
		if image_path.endswith('.tif'):
			filename = image_path.split('/')[-1][:-len(".tif")]
			
			dirname = image_path.split('/')[:-2]
			dirname = '/'.join(dirname)
			framename = filename.split('_')[:-2]
			framename = '_'.join(framename)
			sectionnum = filename.split('_')[-1]			
			image_pathC2 =  dirname + "/insulin/" + framename + "_Insulin_" + sectionnum + ".tif"
			# GT_pathC1 = self.GT_paths+"/" + framename + "_CellMask_" + sectionnum + ".tif"
			# GT_pathC2 = self.GT_paths+"/" + framename + "_NucleiMask_" + sectionnum + ".tif"
			# GT_pathC3 = self.GT_paths+"/" + framename + "_Borders_" + sectionnum + ".tif"
			GT_pathC1 = self.GT_paths+"/" + framename + "_CollagenIV_" + sectionnum + ".tif"
			GT_pathC2 = self.GT_paths+"/" + framename + "_Channel_5_" + sectionnum + ".tif"
			imagec1 = Image.open(image_path).convert("L")
			#imagec2 = Image.open(image_pathC2).convert("L")
			#dummy_channel = Image.new("L", imagec2.size)
			GTc1 = Image.open(GT_pathC1)
			GTc2 = Image.open(GT_pathC2)
			#GTc3 = Image.open(GT_pathC3)
			dummy_channel = Image.new("L", GTc2.size)
			GTc1 = np.array(GTc1)
			GTc2 = np.array(GTc2)
			#GTc3 = np.array(GTc3)
			GTc1[GTc1 > 0] = 1
			GTc2[GTc2 > 0] = 1
			#GTc3[GTc3 > 0] = 1
			GTc1 = Image.fromarray(GTc1).convert("L")
			GTc2 = Image.fromarray(GTc2).convert("L")
			#GTc3 = Image.fromarray(GTc3).convert("L")
			# merging the channels
			#image = Image.merge("RGB", [imagec1, imagec2, dummy_channel])
			image = imagec1

					
			
			# GT_path = self.GT_paths+"/" + filename + ".tif"
			# image = Image.open(image_path).convert("L")
			# GT = Image.open(GT_path)
			# GT = np.array(GT)  
			# GT[GT > 0] = 1
			# GT = Image.fromarray(GT).convert("L")
				
			# add the preceeding and following images as channels
			#image = Image.merge("RGB", (prec_image, image, foll_image))

			# create patches   
			img_patches, GT_patches, bGT_patches  = self._extract_patches(image, GTc1, GTc2)
			# Apply transformations
			img_pro_patches = []
			GT_pro_patches = []
			bGT_pro_patches = []
		


			for img_patch, GT_patch, bGT_patch in zip(img_patches, GT_patches, bGT_patches):
				transform = self.transformation()
				#img_patch = img_patch.squeeze(0)
				#GT_patch = GT_patch.squeeze(0)
				#img_patch = Image.fromarray(img_patch.numpy())
				#GT_patch = Image.fromarray(GT_patch.numpy())
				
				# trasform the patches with augmenting dat
				# img_pro_patch = transform(img_patch)
				# GT_pro_patch = transform(GT_patch)				
				# img_pro_patches.append(img_pro_patch)
				# GT_pro_patches.append(GT_pro_patch)
				
				# transform the patches with augmenting data			
				for T in range(6):
					img_aug, GT_aug, bGT_aug = self.augment_data(img_patch, GT_patch, bGT_patch, T)
					img_pro_patch = transform(img_aug)
					GT_pro_patch = transform(GT_aug)
					bGT_pro_patch = transform(bGT_aug)
					#img_pro_patch = img_aug
					#GT_pro_patch = GT_aug
					img_pro_patches.append(img_pro_patch)
					GT_pro_patches.append(GT_pro_patch)
					bGT_pro_patches.append(bGT_pro_patch)
				
			# Shuffle the patches
			#img_pro_patches, GT_pro_patches = shuffle(img_pro_patches, GT_pro_patches)

			# Convert the lists of patches to tensors
			img_pro = torch.stack(img_pro_patches)
			GT_pro = torch.stack(GT_pro_patches)
			bGT_pro = torch.stack(bGT_pro_patches)
		else:
			print("Invalid file format: {}".format(image_path))
		return img_pro, GT_pro, bGT_pro  

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)


	def transformation(self, mode='train',augmentation_prob=0.4, patch_size=256):
		RotationDegree = [0,90,180,270]
		p_transform = random.random()
		hflip = random.random()
		vflip = random.random()
		transform = []
		if (mode == 'train' and p_transform <= augmentation_prob):
			r = random.randint(0,3)
			Rotation = RotationDegree[r]
			transform.append(T.RandomRotation((Rotation,Rotation)))						
			RotationRange = random.randint(-10,10)
			transform.append(T.RandomRotation((RotationRange,RotationRange)))
			if (hflip < 0.4):
				transform.append(F.hflip)
			if (vflip < 0.4):
				transform.append(F.vflip)
			transform.append(T.RandomResizedCrop(size=(patch_size,patch_size), scale=(0.75, 1.5), ratio=(0.75, 1.33)))
	
		transform.append(T.Resize((self.patch_size, self.patch_size)))	
		#transform.append(T.ToTensor())
		transform = T.Compose(transform)
		return transform


	def augment_data(self, img_patch, GT_patch, bGT_patch, T):
		if (T==0):
			img_aug = F.hflip(img_patch)
			GT_aug = F.hflip(GT_patch)
			bGT_aug = F.hflip(bGT_patch)
		elif (T==1):
			img_aug = F.vflip(img_patch)
			GT_aug = F.vflip(GT_patch)
			bGT_aug = F.vflip(bGT_patch)
		elif (T==2):
			img_aug = F.rotate(img_patch, 90)
			GT_aug = F.rotate(GT_patch, 90)
			bGT_aug = F.rotate(bGT_patch, 90)
		elif (T==3):
			img_aug = F.rotate(img_patch, 180)
			GT_aug = F.rotate(GT_patch, 180)
			bGT_aug = F.rotate(bGT_patch, 180)
		elif (T==4):
			r1 = np.int32(random.random()*0.25*255)
			r2 = np.int32(random.random()*0.25*255)
			c_height = 255-2*r1
			c_width = 255-2*r2
			img_aug = F.resized_crop(img_patch, r1, r2, c_height, c_width, size=(256, 256))
			GT_aug = F.resized_crop(GT_patch, r1, r2, c_height, c_width, size=(256, 256))
			bGT_aug = F.resized_crop(bGT_patch, r1, r2, c_height, c_width, size=(256, 256))
		else:
			img_aug = img_patch
			GT_aug = GT_patch
			bGT_aug = bGT_patch
			
			
		return img_aug, GT_aug, bGT_aug
	

	def augment_data2(self, img_patch, GT_patch, T):
		
		if (T==0):
			img_aug = F.rotate(img_patch, 90)
			GT_aug = F.rotate(GT_patch, 90)
		elif (T==1):
			img_aug = F.rotate(img_patch, 90)
			GT_aug = F.rotate(GT_patch, 90)
			img_aug = F.hflip(img_aug)
			GT_aug = F.hflip(GT_aug)
		elif (T==2):
			img_aug = F.rotate(img_patch, 180)
			GT_aug = F.rotate(GT_patch, 180)
		elif (T==3):
			img_aug = F.rotate(img_patch, 180)
			GT_aug = F.rotate(GT_patch, 180)
			img_aug = F.hflip(img_aug)
			GT_aug = F.hflip(GT_aug)
		elif (T==4):
			img_aug = F.hflip(img_patch)
			GT_aug = F.hflip(GT_patch)
		else:
			img_aug = img_patch
			GT_aug = GT_patch
			
		return img_aug, GT_aug
			
		

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = PatchImageFolder(root = image_path, patch_size=256, stride=128, mode=mode , augmentation_prob=augmentation_prob) 
	data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	return data_loader
