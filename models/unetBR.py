import torch.nn as nn
from .unetBR_parts import UnetResConv2D, UnetResUpConv2D, weights_init_kaiming, ConvNormReLU, Refine
import torch.nn.functional as F


class UnetBR2D(nn.Module):

    def __init__(self, n_classes, norm, is_pool=False):
        super(UnetRes2D, self).__init__()
        print("UnetBoundaryReine2D")
        ch = [16, 32, 64, 128, 256]

        # downsampling
        self.conv1    = UnetResConv2D(1, ch[0], norm, ch[0] / 8)
        self.maxpool1 = nn.MaxPool2d(2) if is_pool else ConvNormReLU(ch[0], ch[0], norm, stride=2)
        self.refine1  = Refine(ch[0])

        self.conv2    = UnetResConv2D(ch[0], ch[1], norm, ch[1] / 8)
        self.maxpool2 = nn.MaxPool2d(2) if is_pool else ConvNormReLU(ch[1], ch[1], norm, stride=2)
        self.refine2  = Refine(ch[1])

        self.conv3    = UnetResConv2D(ch[1], ch[2], norm, ch[2] / 8)
        self.maxpool3 = nn.MaxPool2d(2) if is_pool else ConvNormReLU(ch[2], ch[2], norm, stride=2)
        self.refine3  = Refine(ch[2])

        self.conv4    = UnetResConv2D(ch[2], ch[3], norm, ch[3] / 8)
        self.maxpool4 = nn.MaxPool2d(2) if is_pool else ConvNormReLU(ch[3], ch[3], norm, stride=2)
        self.refine4  = Refine(ch[3])

        self.center   = UnetResConv2D(ch[3], ch[4], norm, ch[4] / 8)
        self.refine_c = Refine(ch[4])
        

        # upsampling
        #                                                     group size
        self.up_concat4 = UnetResUpConv2D(ch[4], ch[3], norm, ch[3] / 8)
        self.up_refine4 = Refine(ch[3])

        self.up_concat3 = UnetResUpConv2D(ch[3], ch[2], norm, ch[2] / 8)
        self.up_refine3 = Refine(ch[2])

        self.up_concat2 = UnetResUpConv2D(ch[2], ch[1], norm, ch[1] / 8)
        self.up_refine2 = Refine(ch[1])

        self.up_concat1 = UnetResUpConv2D(ch[1], ch[0], norm, ch[0] / 8)
        self.up_refine1 = Refine(ch[0])

        self.final = nn.Conv2d(ch[0], 1, 1, 1, 0) 

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.InstanceNorm2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.GroupNorm):
                m.apply(weights_init_kaiming)

    def forward(self, inputs):
        conv1     = self.conv1(inputs)
        maxpool1  = self.maxpool1(conv1)
        br_conv1  = self.refine1(conv1)

        conv2     = self.conv2(maxpool1)
        maxpool2  = self.maxpool2(conv2)
        br_conv2  = self.refine2(conv2)

        conv3     = self.conv3(maxpool2)
        maxpool3  = self.maxpool3(conv3)
        br_conv3  = self.refine3(conv3)

        conv4     = self.conv4(maxpool3)
        maxpool4  = self.maxpool4(conv4)
        br_conv4  = self.refine4(conv4)

        center    = self.center(maxpool4)
        br_center = self.refine_c(center)

        up4       = self.up_concat4(br_conv4, br_center)
        br_up4    = self.up_refine4(up4)
        
        up3       = self.up_concat3(br_conv3, br_up4)
        br_up3    = self.up_refine3(up3)

        up2       = self.up_concat2(br_conv2, br_up3)
        br_up2    = self.up_refine2(up2)

        up1       = self.up_concat1(br_conv1, br_up2)
        br_up1    = self.up_refine1(up1)

        return self.final(up1) 

if __name__ == "__main__":
    import torch
    device = torch.device("cuda")
    input2D = torch.randn([1, 1, 448, 448]).to(device)
    print("input shape : \t", input2D.shape)
    model = UnetRes2D(1, nn.InstanceNorm2d)
    output2D = model(input2D).to(device)
    print("output shape  : \t", output2D.shape)
    from torchsummary import summary
    summary(model, input_size=(1, 448, 448))