import torch
import torch.nn as nn

# reference:
# Learned Spectral Super-Resolution (https://arxiv.org/abs/1703.09470)
# Fully Convolutional DenseNets (https://arxiv.org/abs/1611.09326)

class Dense_Unit(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(Dense_Unit, self).__init__()
        self.unit = nn.Sequential(
            nn.BatchNorm2d(input_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(input_features, output_features, kernel_size=3, padding=1, bias=False),
            nn.Dropout2d(p=p_drop)
        )
    def forward(self, x):
        return self.unit(x)

class DenseBlock_Batch(nn.Module):
    def __init__(self, input_features, growth_rate, negative_slope=0, p_drop=0):
        super(DenseBlock_Batch, self).__init__()
        self.unit1 = Dense_Unit(input_features, growth_rate, negative_slope, p_drop)
        self.unit2 = Dense_Unit(input_features+growth_rate, growth_rate, negative_slope, p_drop)
        self.unit3 = Dense_Unit(input_features+2*growth_rate, growth_rate, negative_slope, p_drop)
        self.unit4 = Dense_Unit(input_features+3*growth_rate, growth_rate, negative_slope, p_drop)
    def forward(self, x):
        unit1 = self.unit1(x)
        stack = torch.cat((x,unit1), 1)
        unit2 = self.unit2(stack)
        stack = torch.cat((stack,unit2), 1)
        unit3 = self.unit3(stack)
        stack = torch.cat((stack,unit3), 1)
        unit4 = self.unit4(stack)
        # output channels: 4*growth_rate
        return torch.cat((unit1,unit2,unit3,unit4), 1)

class Down_Block(nn.Module):
    def __init__(self, features, negative_slope=0, p_drop=0):
        super(Down_Block, self).__init__()
        self.down = nn.Sequential(
            nn.BatchNorm2d(features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(features, features, kernel_size=1, padding=0, bias=False),
            nn.Dropout2d(p=p_drop),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.down(x)

class Up_Block(nn.Module):
    def __init__(self, features, growth_rate, negative_slope=0, p_drop=0):
        super(Up_Block, self).__init__()
        self.up = nn.Sequential(
            DenseBlock_Batch(features, growth_rate, negative_slope, p_drop),
            nn.PixelShuffle(upscale_factor=2)
        )
    def forward(self, x):
        return self.up(x)

class Model_Ref(nn.Module):
    def __init__(self, input_features, pre_features, output_features, db_growth_rate, negative_slope=0, p_drop=0):
        super(Model_Ref, self).__init__()
        self.features = [
            pre_features,
            pre_features+4*db_growth_rate,
            pre_features+2*4*db_growth_rate,
            pre_features+3*4*db_growth_rate,
            pre_features+4*4*db_growth_rate,
            pre_features+5*4*db_growth_rate,
            pre_features+(5*4+1)*db_growth_rate,
            pre_features+(4*4+1)*db_growth_rate,
            pre_features+(3*4+1)*db_growth_rate,
            pre_features+(2*4+1)*db_growth_rate,
            pre_features+(4+1)*db_growth_rate
        ]
        self.conv = nn.Conv2d(input_features, self.features[0], kernel_size=3, padding=1, bias=False)
        self.dense1 = DenseBlock_Batch(self.features[0], db_growth_rate, negative_slope=negative_slope, p_drop=p_drop)
        self.down1 = Down_Block(self.features[1], negative_slope, p_drop)
        self.dense2 = DenseBlock_Batch(self.features[1], db_growth_rate, negative_slope=negative_slope, p_drop=p_drop)
        self.down2 = Down_Block(self.features[2], negative_slope, p_drop)
        self.dense3 = DenseBlock_Batch(self.features[2], db_growth_rate, negative_slope=negative_slope, p_drop=p_drop)
        self.down3 = Down_Block(self.features[3], negative_slope, p_drop)
        self.dense4 = DenseBlock_Batch(self.features[3], db_growth_rate, negative_slope=negative_slope, p_drop=p_drop)
        self.down4 = Down_Block(self.features[4], negative_slope, p_drop)
        self.dense5 = DenseBlock_Batch(self.features[4], db_growth_rate, negative_slope=negative_slope, p_drop=p_drop)
        self.down5 = Down_Block(self.features[5], negative_slope, p_drop)
        self.bottleneck = DenseBlock_Batch(self.features[5], db_growth_rate, negative_slope=negative_slope, p_drop=p_drop)
        self.up1 = nn.PixelShuffle(upscale_factor=2)
        self.up2 = Up_Block(self.features[6], db_growth_rate, negative_slope, p_drop)
        self.up3 = Up_Block(self.features[7], db_growth_rate, negative_slope, p_drop)
        self.up4 = Up_Block(self.features[8], db_growth_rate, negative_slope, p_drop)
        self.up5 = Up_Block(self.features[9], db_growth_rate, negative_slope, p_drop)
        self.final = nn.Sequential(
            DenseBlock_Batch(self.features[10], db_growth_rate, negative_slope=negative_slope, p_drop=p_drop),
            nn.Conv2d(4*db_growth_rate, output_features, kernel_size=3, padding=1, bias=False)
        )
    def forward(self, x):
        x = self.conv(x)
        ## downsample path
        ## repreat 5 times: dense block + skip connection + downsample
        # 1. output features: pre_features + 4*db_growth_rate
        down_cat1 = torch.cat((x, self.dense1(x)), 1)
        x = self.down1(down_cat1)
        # 2. output features: pre_features + 2*4*db_growth_rate
        down_cat2 = torch.cat((x, self.dense2(x)), 1)
        x = self.down2(down_cat2)
        # 3. output features: pre_features + 3*4*db_growth_rate
        down_cat3 = torch.cat((x, self.dense3(x)), 1)
        x = self.down3(down_cat3)
        # 4. output features: pre_features + 4*4*db_growth_rate
        down_cat4 = torch.cat((x, self.dense4(x)), 1)
        x = self.down4(down_cat4)
        # 5. output features: pre_features + 5*4*db_growth_rate
        down_cat5 = torch.cat((x, self.dense5(x)), 1)
        x = self.down5(down_cat5)
        ## bottleneck
        # output features: pre_features + 6*4*db_growth_rate
        x = self.bottleneck(x)
        ## upsample path
        ## repreat 5 times: subpixel upsample + skip connection + dense block
        # 1. output features: pre_features+(5*4+1)*db_growth_rate
        x = self.up1(x)
        x = torch.cat((down_cat5, x), 1)
        # 2. output features: pre_features+(4*4+1)*db_growth_rate
        x = self.up2(x)
        x = torch.cat((down_cat4, x), 1)
        # 3. output features: pre_features+(3*4+1)*db_growth_rate
        x = self.up3(x)
        x = torch.cat((down_cat3, x), 1)
        # 4. output features: pre_features+(2*4+1)*db_growth_rate
        x = self.up4(x)
        x = torch.cat((down_cat2, x), 1)
        # 5. output features: pre_features+(4+1)*db_growth_rate
        x = self.up5(x)
        x = torch.cat((down_cat1, x), 1)
        ## mapping to the output
        x = self.final(x)
        return x
