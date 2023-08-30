import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))

class DegradationEncoder(nn.Module):
    def __init__(self):
        super(DegradationEncoder, self).__init__()

        self.E_pre = ResBlock(in_feat=3, out_feat=64, stride=1)
        self.E = nn.Sequential(
            ResBlock(in_feat=64, out_feat=128, stride=2),
            ResBlock(in_feat=128, out_feat=256, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.pred_list = nn.ModuleList()
        self.mask_list = nn.ModuleList()
        layers = [1,2,4,8]
        for num in layers:
            self.pred_list.append(
                nn.Sequential(
                nn.Linear(256, 256),
                nn.LeakyReLU(0.1, True),
                nn.Linear(256,num))
                )
            self.mask_list.append(
                nn.Sequential(
                    nn.Linear(256, 256),
                    nn.LeakyReLU(0.1, True),
                    nn.Linear(256, num),
                    nn.Sigmoid()
                )
            )

    def forward(self, x, pos=None):
        inter = self.E_pre(x)
        fea = self.E(inter).squeeze(-1).squeeze(-1)
        out_mask = []
        out_pred = []
        for idx in range(pos):
            temp_mask = self.mask_list[idx](fea)
            out_mask.append(temp_mask)
            temp_gred = self.pred_list[idx](fea)
            out_pred.append(temp_gred)
        return out_mask, out_pred




class multiLinear(nn.Module):
    def __init__(self, in_channels = 3, out_channels=3):
        super(multiLinear, self).__init__()
        self.linear_list = nn.ModuleList()
        self.linear_list.append(nn.Linear(1, out_channels))
        layers = [2,4,8]
        for num in layers:
            self.linear_list.append(
                nn.Linear(num, out_channels)
            )
    def forward(self, x):
        out = None
        for idx in range(len(x)):
            layer = self.linear_list[idx]
            out = out + layer(x[idx]) if out is not None else layer(x[idx])
        return out

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = (1. / torch.sqrt(var + eps)) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)



class CLayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(-1, C, 1, 1) * y + bias.view(-1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(-1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = (1. / torch.sqrt(var + eps)) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2), grad_output.sum(dim=3).sum(dim=2), None

class CLayerNorm2d(nn.Module):

    def __init__(self, channels, rep_c,eps=1e-6):
        super(CLayerNorm2d, self).__init__()
        self.eps = eps
        self.weight = multiLinear(rep_c,channels)
        self.bias = multiLinear(rep_c,channels)
    def forward(self, x, label):
        weights = self.weight(label)
        bias = self.bias(label)
        return CLayerNormFunction.apply(x, weights, bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inputs):
        inp,_ = inputs

        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        #y = inp + x
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        #return y + x
        return y + x * self.gamma

class CSimpleGate(nn.Module):
    def __init__(self, c, rep_c):
        super().__init__()
        self.embedding = multiLinear(rep_c, c)
    def forward(self, x, label):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2 * self.embedding(label).unsqueeze(2).unsqueeze(3)

class LabelNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        #self.weights = nn.Embedding(2, c)
        self.bias = nn.Embedding(2, c)
    def forward(self, x, label):
        return x + self.bias(label).unsqueeze(2).unsqueeze(3)

class FTBlock(nn.Module):
    def __init__(self, c, rep_c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg_1 = CSimpleGate(c, rep_c)
        self.sg_2 = CSimpleGate(c, rep_c)
        #self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = CLayerNorm2d(c, rep_c)
        self.norm2 = CLayerNorm2d(c, rep_c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = multiLinear(rep_c, c)
        self.gamma = multiLinear(rep_c, c)
    def forward(self, inputs):
        inp, label = inputs

        x = inp
        x =  self.norm1(x, label)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg_1(x, label)
        #x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta(label).unsqueeze(2).unsqueeze(3)
        x = self.norm2(y,label)
        x = self.conv4(x)
        x = self.sg_2(x, label)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma(label).unsqueeze(2).unsqueeze(3)

class AMIRNet(nn.Module):

    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1]):
        super().__init__()
        rep_c = 15
        self.resencoder = DegradationEncoder()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            if num == 1:
                self.encoders.append( 
                    nn.Sequential(
                        *[FTBlock(chan, rep_c) for _ in range(num)]
                    )
                )
            else:
                self.encoders.append(
                    nn.Sequential(
                        *[FTBlock(chan, rep_c) for _ in range(num)]
                    )
                )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
    def forward(self, inp, pos = 4):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        mask, pred = self.resencoder(inp, pos)

        label = []
        for each_mask, each_pred in zip(mask, pred):
            label.append(F.softmax(each_mask,dim=1) * each_pred)
        
        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            for each_block in encoder:
                x = each_block((x, label))
            encs.append(x)
            x = down(x)

        for latent_encoder in self.middle_blks:
            x = latent_encoder((x, label))
        #x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder((x, label))

        x = self.ending(x)
        x = x + inp[:,:3,:,:]
        return x[:, :, :H, :W], (mask, pred)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = AMIRNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
