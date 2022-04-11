import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torchvision.models.resnet import Bottleneck

YAW_MIN,YAW_MAX     = -100,100
PITCH_MIN,PITCH_MAX = -100,20
ROLL_MIN,ROLL_MAX   = -100,100

class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins, final_size):
        self.inplanes = 64
        super(Hopenet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AdaptiveAvgPool2d(final_size)
        self.fc_yaw = nn.Linear(512 * block.expansion*final_size**2, num_bins[0])
        self.fc_pitch = nn.Linear(512 * block.expansion*final_size**2, num_bins[1])
        self.fc_roll = nn.Linear(512 * block.expansion*final_size**2, num_bins[2])

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        idx_tensor_yaw = torch.arange(YAW_MIN,YAW_MAX,(YAW_MAX-YAW_MIN)/num_bins[0])
        idx_tensor_pitch = torch.arange(PITCH_MIN,PITCH_MAX,(PITCH_MAX-PITCH_MIN)/num_bins[1])
        idx_tensor_roll = torch.arange(ROLL_MIN,ROLL_MAX,(ROLL_MAX-ROLL_MIN)/num_bins[2])
        self.register_buffer("idx_tensor_yaw",idx_tensor_yaw)
        self.register_buffer("idx_tensor_pitch",idx_tensor_pitch)
        self.register_buffer("idx_tensor_roll",idx_tensor_roll)
        self.register_buffer("yaw_related",torch.Tensor([YAW_MIN,YAW_MAX,num_bins[0]]))
        self.register_buffer("pitch_related",torch.Tensor([PITCH_MIN,PITCH_MAX,num_bins[1]]))
        self.register_buffer("roll_related",torch.Tensor([ROLL_MIN,ROLL_MAX,num_bins[2]]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll

class ResNet(nn.Module):
    # ResNet for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_angles = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_angles(x)
        return x

class AlexNet(nn.Module):
    # AlexNet laid out as a Hopenet - classify Euler angles in bins and
    # regress the expected value.
    def __init__(self, num_bins):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc_yaw = nn.Linear(4096, num_bins)
        self.fc_pitch = nn.Linear(4096, num_bins)
        self.fc_roll = nn.Linear(4096, num_bins)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        return yaw, pitch, roll



class FrameAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, seq_len, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.seq_len = seq_len  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
    #     self.relative_position_bias_table = nn.Parameter(
    #         torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

    #     # get pair-wise relative position index for each token inside the window
    #     coords_h = torch.arange(self.window_size[0])
    #     coords_w = torch.arange(self.window_size[1])
    #     coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    #     coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    #     relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    #     relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    #     relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
    #     relative_coords[:, :, 1] += self.window_size[1] - 1
    #     relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    #     relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    #     self.register_buffer("relative_position_index", relative_position_index)
        # trunc_normal_(self.relative_position_bias_table, std=.02)

        self.abs_position_encoding = nn.Parameter(torch.zeros(num_heads,seq_len,dim//3)) # 3 N C
        trunc_normal_(self.abs_position_encoding,std=0.02)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # import pdb;pdb.set_trace()
        # B 3 heads Seq C
        # 这个地方问题: C必须能被num_heads整除
        # dim: 3,B,num_heads,N,C
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple), of shape: B,num_heads,N,C

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # B heads Seq Seq

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.seq_len[0] * self.seq_len[1], self.seq_len[0] * self.seq_len[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #relative_position_bias是一个跟具体的数据无关的特征, 但是它是nn.parameter,所以也是一个学习参数
        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # heads N C
        v = v + self.abs_position_encoding.unsqueeze(0)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Hopenet_multi_frame(nn.Module):
    def __init__(self,layers,num_bins,seq_len):
        super(Hopenet_multi_frame,self).__init__()
        self.backbone = Hopenet(Bottleneck,layers,[num_bins]*3,1)
        self.attention_layer = FrameAttention(num_bins,seq_len,3)

    def forward(self,x):
        B,N,C,H,W = x.shape
        x = x.view(B*N,C,H,W)
        yaw,pitch,roll = self.backbone(x)

        yaw = yaw.view(B,N,-1)
        pitch = pitch.view(B,N,-1)
        roll = roll.view(B,N,-1)

        yaw = self.attention_layer(yaw)
        pitch = self.attention_layer(pitch)
        roll = self.attention_layer(roll)

        return yaw,pitch,roll
