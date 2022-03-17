"""
이전 ver1과 다른점은 Linear대신 convolution을 사용한것
마지막은 3개의 feature 각각 return
"""
import torch
from torch import nn


class HFM(nn.Module):
    def __init__(self, features, WH=32, G=8, r=2, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(HFM, self).__init__()
        # 이미지 벡터 압축 비율
        d = max(int(features/r), L)
        self.features = features
        
        # LH, HL, HH 채널 concat했을때 코드
        # 채널을 3c에서 c로 맞추기 위해 1*1 convolution 사용
        """
        self.conv = nn.Sequential(
            nn.Conv2d(3*features, features, kernel_size=1, stride=stride, padding=1, groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        )
        """
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 이미지 벡터 압축 Linear층
        #self.fc = nn.Linear(features, d)
        self.fc = nn.Sequential(nn.Conv2d(features, d, 1, padding=0, bias=False), nn.PReLU())

        # 각 LH, HL, HH 에 곱해질 벡터 --> LH, HL, HH에서 의미있는 피쳐만 뽑길 기대......
        self.fcs = nn.ModuleList([])
        #for i in range(3):
        #    self.fcs.append(nn.Linear(d, features))
        for i in range(3):
                self.fcs.append(nn.Conv2d(d, features, kernel_size=1, stride=1,bias=False))

        # 피쳐를 선택하기 위해 softmax거침 --> 위 __init__에 대한 내용을 모르면 논문 or 깃허브에 그림 참고!!
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x1, x2, x3):
        # concat version --> 채널 맞추기 위해 1*1convolution 연산
        """
        concat_x = torch.cat(x_list, dim=1)
        fea_U = self.conv(concat_x)
        """
        
        """
        # sum version
        x_list = torch.cat(x_list, dim=1)
        fea_U = torch.sum(x_list, dim=1)
        """
        
        # 그림에 가운데 모든 피쳐를 합치는 부분
        #print(x1.shape)
        batch = x1.shape[0]
        fea_U = (x1+x2+x3).cuda()
        #GAP
        # fea_s = self.gap(fea_U).squeeze_()
        #fea_s = fea_U.mean(-1).mean(-1)
        fea_s = self.avg_pool(fea_U)
        #print('1. fea_s: {}'.format(fea_s.shape))
        # 이미지 벡터 압축
        fea_z = self.fc(fea_s)
        #print('2. fea_z: {}'.format(fea_z.shape))
        # 이미지 벡터 원래 차원으로 늘림
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z)
            #print('3. vector: {}'.format(vector.shape))
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=0)
                
        #print('4. attention_vectors: {}'.format(attention_vectors.shape))
        # like se-block version
        """
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (x_list * attention_vectors).sum(dim=1)
        """
        # 각 LH, HL, HH에 피쳐 select벡터 곱해줌
        # softmax version
        attention_vectors = self.softmax(attention_vectors)
        f1 = x1 * attention_vectors[:batch]
        f2 = x2 * attention_vectors[batch:2*batch]
        f3 = x3 * attention_vectors[2*batch:]
        return f1, f2, f3