import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            #nn.LayerNorm(ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),       
            #nn.LayerNorm(ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    #nn.LayerNorm(ch_out),
            nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    #nn.LayerNorm(ch_out),
            nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.size() != x1.size():
            x1 = F.interpolate(x1, size=g1.shape[2:])
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        if psi.size() != x.size():
            x = F.interpolate(x, size=psi.shape[2:])

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(U_Net,self).__init__()
        
        #self.normalizer = nn.LayerNorm([3, 768, 768])
        self.normalizer = nn.LayerNorm([1, 256, 256])

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.Conv6 = conv_block(ch_in=1024,ch_out=2048)
        
        #self.Up6 = up_conv(ch_in=2048,ch_out=1024)
        self.Up6 = up_conv(ch_in=2048,ch_out=1024)
        self.Up_conv6 = conv_block(ch_in=2048, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.BottleNeck = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.Att1 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Att2 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Att3 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Att4 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Att5 = Attention_block(F_g=1024,F_l=1024,F_int=512)

        

    def forward(self,x):
        # Normalization
        #x = self.normalizer(x)
        
              
        
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        x6 = self.Maxpool(x5)
        x6 = self.Conv6(x6)

        # bottleneck
        x6 = self.BottleNeck(x6)
        
        # decoding + concat path
        d6 = self.Up6(x6)
        d6 = torch.nn.functional.interpolate(d6, size=x5.shape[2:]) 
        g5 = self.Att5(g=d6,x=x5)
        d6 = torch.cat((g5,d6),dim=1)
        #d6 = torch.cat((x5,d6),dim=1)
        # gp5 = self.Att5(g=d6,x=xp5)
        # gf5 = self.Att5(g=d6,x=xf5)
        # d6 = torch.cat((gp5,d6,gf5),dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(x5)
        d5 = torch.nn.functional.interpolate(d5, size=x4.shape[2:])
        g4 = self.Att4(g=d5,x=x4)
        d5 = torch.cat((g4,d5),dim=1)
        #d5 = torch.cat((x4, d5), dim=1)  
        # gp4 = self.Att4(g=d5,x=xp4) 
        # gf4 = self.Att4(g=d5,x=xf4)
        # d5 = torch.cat((gp4,d5,gf4),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.nn.functional.interpolate(d4, size=x3.shape[2:])
        g3 = self.Att3(g=d4,x=x3)
        d4 = torch.cat((g3,d4),dim=1)
        #d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.nn.functional.interpolate(d3, size=x2.shape[2:])
        #g2 = self.Att2(g=d3,x=x2)
        #d3 = torch.cat((g2,d3),dim=1)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.nn.functional.interpolate(d2, size=x1.shape[2:])
        #g1 = self.Att1(g=d2,x=x1)
        #d2 = torch.cat((g1,d2),dim=1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1, x6, d6, d5, d4
        









