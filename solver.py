import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
from network_standard import UNet
import csv
import torch.nn as nn
from PIL import Image
from tqdm import tqdm 
import matplotlib.pyplot as plt

from torchvision.transforms import ToPILImage

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		#self.criterion2 = self.diceLoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()
		
		# addtional 
		self.BottleNeck_size = 2048
	
	def diceLoss(self, x, y):
		intersection = torch.sum(x * y) + 1e-7
		union = torch.sum(x) + torch.sum(y) + 1e-7
		return 1 - 2 * intersection / union
		
			
	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=1,output_ch=2)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=1,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=1,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=1,output_ch=1,t=self.t)
		else:
			self.unet = UNet(in_channels=2,out_channels=3)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)
		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img
	
	
	def binning(self, x):
		batch_size, channels, height, width = x.size()
		x = x.view(batch_size, channels, height // 2, 2, width // 2, 2)
		x = x.float()
		x = x.mean(dim=(3, 5))
		return x
	
	def bin_gt(self, gt, i):
		for i in range(i):
			gt = self.binning(gt)
		return gt
		
	def check_bin(self, bn_gt, original_gt, i):
		# create the output path
		output_path = './dataset/binned_mask/'
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		stretching = nn.Upsample(scale_factor=16)	
		mask_stretched = stretching(bn_gt)
		for n in range(mask_stretched.shape[0]):
			mask = mask_stretched[n, 0, :, :]
			mask = mask.squeeze(0)
			mask = mask.squeeze(0)
			mask = mask - mask.min()
			mask = mask / mask.max()
			mask = ((mask) * 255).byte()
			mask = mask.cpu().numpy()
			mask = Image.fromarray(mask, mode='L')
			k = i*64 + n
			mask.save(os.path.join(output_path,"maskgt_"+str(k)+".tif"))
			gt = original_gt[n, 0, :, :]
			gt = gt.squeeze(0)
			gt = gt.squeeze(0)
			gt = gt - gt.min()
			gt = gt / gt.max()
			gt = ((gt) * 255).byte()
			gt = gt.cpu().numpy()
			gt = Image.fromarray(gt, mode='L')
			k = i*64 + n
			gt.save(os.path.join(output_path,"gt_"+str(k)+".tif"))

	def check_images(self, images, GT):
		
		# create the output path
		output_path = './dataset/check_images/'
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		images_path = './dataset/check_images/images/'
		if not os.path.exists(images_path):
			os.makedirs(images_path)
		gt_path = './dataset/check_images/gt/'
		if not os.path.exists(gt_path):
			os.makedirs(gt_path)	
		to_pil = ToPILImage()	
		for n in range(images.shape[0]):
			image = images[n, 0, :, :]
			image = image.squeeze(0)
			#image = image.squeeze(0)
			image = image - image.min()
			image = image / image.max()
			image = ((image) * 255).byte()
			#image = image.cpu().numpy()
			#image = Image.fromarray(image, mode='L')
			image = to_pil(image)
			image.save(os.path.join(images_path,"image_"+str(n)+".png"))
			mask = GT[n, 0, :, :]
			mask = mask.squeeze(0)
			#mask = mask.squeeze(0)
			mask = mask - mask.min()
			mask = mask / mask.max()
			mask = ((mask) * 255).byte()
			#mask = mask.cpu().numpy()
			#mask = Image.fromarray(mask, mode='L')
			mask = to_pil(mask)
			mask.save(os.path.join(gt_path,"gt_"+str(n)+".png"))
			
	def visualiz_processing(self, images, GTs, SR_probs, BN_img, fig, axes):
		 
		for i in range(3):
			image = images[i].squeeze(0).byte().cpu().numpy()
			axes[i,0].clear()
			axes[i,0].imshow(image)
			axes[i,0].set_title("image")
			axes[i,0].axis('off')
			axes[i,0].set_aspect('equal')
			
		for i in range(3):
			GT = GTs[i].squeeze(0).byte().cpu().numpy()
			GT[GT>0.5] = 255
			axes[i,1].clear()
			axes[i,1].imshow(GT)
			axes[i,1].set_title("GT")
			axes[i,1].axis('off')
			axes[i,1].set_aspect('equal')
			
		for i in range(3): 
			SR_prob = SR_probs[i].detach().cpu().numpy()	
            # check images and GT
			SR_prob[SR_prob>=0.5] = 255
			
			axes[i,2].clear()
			axes[i,2].imshow(SR_prob.astype(np.uint8))
			axes[i,2].set_title("Prediction")
			axes[i,2].axis('off')
			axes[i,2].set_aspect('equal')
			axes[i,3].clear()
			axes[i,3].imshow(SR_prob.astype(np.uint8))
			axes[i,3].set_title("predication")
			axes[i,3].axis('off')
			axes[i,3].set_aspect('equal')
			

		for i in range(3):
			BN = BN_img[i].detach().cpu().numpy()
			BN = BN.transpose(1, 2, 0)  # Ensure correct shape (height, width, channels)
			BN = BN * 255
			axes[i,4].clear()
			axes[i,4].imshow(BN)
			axes[i,4].set_title("BottleNeck")
			axes[i,4].axis('off')
			axes[i,4].set_aspect('equal')
	
		plt.pause(0.001)
		plt.draw()
                    
		

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		
		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			startTime = time.time()
			lr = self.lr
			best_unet_score = 0.
			best_unet = None
			best_epoch = 0
			
			plt.ion()
			fig, axes = plt.subplots(3, 5, figsize=(25,25)) 
			best_loss = 1000
			
			# test data loading 
			#images_patches, GT_patches = self.train_loader.dataset.__getitem__(6)

			for epoch in tqdm(range(self.num_epochs)):
			#for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				val_loss = 0
				test_loss = 0
				
				
				
				print("epoch", epoch+1)
				# check data loader
				images_patches, images_aux_patches, GT_patches, nGT_patches, bGT_patches = self.train_loader.dataset.__getitem__(6)


				# claculate guided losses
				conv4 = nn.Conv2d(int(self.BottleNeck_size/2), 1, kernel_size=1).to(self.device)
				conv3 = nn.Conv2d(int(self.BottleNeck_size/4), 1, kernel_size=1).to(self.device)
				conv2 = nn.Conv2d(int(self.BottleNeck_size/8), 1, kernel_size=1).to(self.device)
				
				for i, (images_patches, images_aux_patches, GT_patches, nGT_patches, bGT_patches) in enumerate(self.train_loader):
					# train for each patch
					torch.save(self.unet.state_dict(), unet_path)
					
					for i in range(images_patches.size(1)):
						images = images_patches[:, i, :, :, :]
						images_aux = images_aux_patches[:, i, :, :, :]
						GT = GT_patches[:, i, :, :, :]
						nGT = nGT_patches[:, i, :, :, :]
						bGT = bGT_patches[:, i, :, :, :]
						images = images.to(self.device)
						images_aux = images_aux.to(self.device)
						images2c = torch.cat((images,images_aux),dim=1)
						GT = GT.to(self.device)
						nGT = nGT.to(self.device)
						bGT = bGT.to(self.device)
						
					
						# Standard U-Net
						SR, btnk = self.unet(images2c)
						SR_probs = F.sigmoid(SR)
						convbn = nn.Conv2d(1024, 1, kernel_size=1).to(self.device)
						btnk = convbn(btnk.to(self.device))
						btnk = F.sigmoid(btnk)
						gt_btlnekc = self.bin_gt(GT, 4)
						gt_btlnekn = self.bin_gt(nGT, 4)
						gt_btlnekb = self.bin_gt(bGT, 4)
						gt_btlnek = torch.cat((gt_btlnekc, gt_btlnekn, gt_btlnekb), dim=1)
						conv_gtbn = nn.Conv2d(3, 1, kernel_size=1).to(self.device)
						gt_btlnek = conv_gtbn(gt_btlnek)
						loss_btlnek = self.diceLoss(btnk.view(btnk.size(0), -1), gt_btlnek.view(gt_btlnek.size(0), -1))
					

						

						# # Weighting boundary channel
						SR_cell = SR_probs[:, 0, :, :]
						SR_neicli = SR_probs[:, 1, :, :]
						SR_boundary = SR_probs[:, 2, :, :]
						loss_cell = self.diceLoss(SR_cell.view(SR_cell.size(0), -1), GT.view(GT.size(0), -1))
						loss_neicli = self.diceLoss(SR_neicli.view(SR_neicli.size(0), -1), nGT.view(nGT.size(0), -1))
						loss_boundary = self.diceLoss(SR_boundary.view(SR_boundary.size(0), -1), bGT.view(bGT.size(0), -1))
						loss_mask = 0.15*loss_cell + 0.35*loss_neicli + 0.5*loss_boundary
					

											
						# calculate full loss
						loss = 0.5*loss_mask + 0.5*loss_btlnek #+ 0.15*guid_loss
						epoch_loss += 0.5*loss_mask.item() + 0.5*loss_btlnek.item() #+ 0.15*guid_loss.item()
					

						# Backprop + optimize
						self.reset_grad()
						loss.backward()
						self.optimizer.step()

											

						# Visualization 
						if (images.size(0) >= 3):
							imagesV = [images[0, :, :, :], images_aux[1, :, :, :], images[2, :, :, :]]  
							GTsV = [GT[0, :, :], nGT[1, :, :], bGT[2, :, :]]
							SR_probsV = [SR_cell[0, :, :], SR_neicli[1, :, :], SR_boundary[2, :, :]]
							#img_probsV = [img_probs[0, 0, :, :], img_probs[1, 0, :, :], img_probs[2, 0, :, :]]
							btnksV = [btnk[0, :, :, :], btnk[1, :, :, :], btnk[2, :, :, :]]
							self.visualiz_processing(imagesV, GTsV, SR_probsV, btnksV, fig, axes)
							#plt.tight_layout()
							#plt.show()


				
				# Print the log info
				print("[INFO] EPOCH: {}/{}".format(epoch+1, self.num_epochs))
				print("Train loss: {:.6f}".format(epoch_loss))
				
							

				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				
				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length=0
				for i, (imageval, aux_imageval, GTval, nGTval, bGTval) in enumerate(self.valid_loader):
					for i in range(imageval.size(1)):
						images = imageval[:, i, :, :, :]
						images_aux = aux_imageval[:, i, :, :, :]
						GT = GTval[:, i, :, :, :]
						nGT = nGTval[:, i, :, :, :]
						bGT = bGTval[:, i, :, :, :]
						images = images.to(self.device)
						images_aux = images_aux.to(self.device)
						GT = GT.to(self.device)
						nGT = nGT.to(self.device)
						bGT = bGT.to(self.device)
						imagesc2 = torch.cat((images,images_aux),dim=1)
						GT3c = torch.cat((GT, nGT, bGT), dim=1)
						
						# # Standard U-Net validation
						SR, _  = self.unet(imagesc2)
						SR_probs= F.sigmoid(SR)

						SR_flat = SR_probs.view(SR_probs.size(0),-1)	
						GT_flat = GT3c.view(GT3c.size(0),-1)
									
						loss_val = self.diceLoss(SR_flat,GT_flat)
						#loss = self.criterion(SR_flat,GT_flat)
						
						val_loss += loss_val.item()

						
				print('[Validation] loss: %.4f'%(val_loss))
						
				unet_path = os.path.join(self.model_path, '%s-%.4f-%d-%.4f.pkl' %(self.model_type,self.lr,epoch,val_loss))
				torch.save(self.unet.state_dict(),unet_path)
				# Save Best U-Net model
				if (val_loss < best_loss):
					best_loss = val_loss
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					unet_best_path = os.path.join(self.model_path, '%s-%.4f-%d-%.4f-best.pkl' %(self.model_type,self.lr,best_epoch,best_loss))
					torch.save(best_unet,unet_best_path)
			
			

	