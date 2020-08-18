import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.xception import default_cfgs, SeparableConv2d, Block, Xception
from training.tools.model_utils import load_pretrained

__all__ = ['xception_map']


class XceptionMap(Xception):

    def __init__(self, templates, num_classes=1000, in_chans=3, **kwargs):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(XceptionMap, self).__init__(num_classes=num_classes, in_chans=in_chans, **kwargs)

        self.templates = templates
        # Reg Map
        self.map = SeparableConv2d(728, 1, 3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # MAM Map
        self.bnmap = nn.BatchNorm2d(728)
        self.convmap = SeparableConv2d(728, 10, 3, 1, 1)
        self.convmap2 = SeparableConv2d(10, 1, 3, 1, 1)
        self.linearmap = nn.Linear(3610, 10)
        self.softmax = nn.Softmax(dim=1)
        self.avgpool2d = nn.AvgPool2d(19, stride=1)
        self.map_conv1 = Block(728, 364, 2, 2, start_with_relu=True, grow_first=False)
        self.map_linear = nn.Linear(364, 10)

    def mask_reg(self, x):
        mask = self.map(x)
        mask = self.sigmoid(mask)
        x = x * mask
        return x, mask, 0

    def mask_template(self, x):
        vec = self.map_conv1(x)
        vec = self.relu(vec)
        vec = F.adaptive_avg_pool2d(vec, (1, 1))
        vec = vec.view(vec.size(0), -1)
        vec = self.map_linear(vec)
        mask = torch.mm(vec, self.templates.reshape(10, 361))
        mask = mask.reshape(x.shape[0], 1, 19, 19)
        x = x * mask
        return x, mask, vec

    def mask_pca_template(self, x):
        fe = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        fe = torch.transpose(fe, 1, 2)
        mu = torch.mean(fe, 2, keepdim=True)
        fea_diff = fe - mu

        cov_fea = torch.bmm(fea_diff, torch.transpose(fea_diff, 1, 2))
        B = self.templates.reshape(1, 10, 361).repeat(x.shape[0], 1, 1)
        D = torch.bmm(torch.bmm(B, cov_fea), torch.transpose(B, 1, 2))
        eigen_value, eigen_vector = D.symeig(eigenvectors=True)
        index = torch.tensor([9]).cuda()
        eigen = torch.index_select(eigen_vector, 2, index)

        vec = eigen.squeeze(-1)
        mask = torch.mm(vec, self.templates.reshape(10, 361))
        mask = mask.reshape(x.shape[0], 1, 19, 19)
        x = x * mask
        return x, mask, vec

    def forward_features(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x, mask, vec = self.mask_reg(x)
        # x, mask, vec = self.mask_template(x)
        # x, mask, vec = self.mask_pca_template(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        # Exit flow
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x, mask, vec

    def forward(self, x):
        x, mask, vec = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x, mask, vec


def xception_map(pretrained=False, num_classes=1000, in_chans=3, templates=0, **kwargs):
    default_cfg = default_cfgs['xception']
    model = XceptionMap(templates, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans, strict=False)
    return model
