import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as func
import util
from torch.nn.modules.utils import _pair

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class MLP_Att_Predict(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Att_Predict, self).__init__()
        self.fc = nn.Linear(opt.resSize, opt.attSize)
        #self.sigmoid =  nn.Sigmoid()
    def forward(self, x):
        o = self.fc(x)
        return o

class MLP_Class_Predict(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Class_Predict, self).__init__()
        self.fc = nn.Linear(opt.resSize, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class MLP_Att_Binary(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Att_Binary, self).__init__()
        self.fc1_img_com = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_com_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_com = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_com_bn = nn.BatchNorm1d(opt.fc2_size)
        # binary
        self.fc1_img_binary = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_binary_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_binary = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_binary_bn = nn.BatchNorm1d(opt.fc2_size)
        # image concat
        #self.fc_img_concat = nn.Linear(opt.fc2_size*2, opt.fc2_size)
        #self.fc_img_concat_bn = nn.BatchNorm1d(opt.fc2_size)
        # Att
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        self.fc_cls = nn.Linear(opt.fc2_size, nclass)
        self.fc_binary = nn.Linear(opt.fc2_size, opt.attSize)
        #self.fc_binary_bn = nn.BatchNorm1d(opt.attSize)
        #
        #self.fc_img_reconst = nn.Linear(nclass, opt.resSize)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res, att):
        img_embed_com = self.fc1_img_com_bn(self.fc1_img_com(res))
        img_embed_com = self.relu(img_embed_com)
        img_embed_com = self.dropout(img_embed_com)
        img_embed_com = self.fc2_img_com_bn(self.fc2_img_com(img_embed_com))
        #
        img_embed_binary = self.fc1_img_binary_bn(self.fc1_img_binary(res))
        img_embed_binary = self.relu(img_embed_binary)
        img_embed_binary = self.dropout(img_embed_binary)
        img_embed_binary = self.fc2_img_binary_bn(self.fc2_img_binary(img_embed_binary))
        # fuse two embed
        img_embed_concat = img_embed_com + img_embed_binary
        #img_embed_concat =  torch.cat((img_embed_com, img_embed_binary), 1)
        #img_embed_concat = self.relu(img_embed_concat)
        #img_embed_concat = self.fc_img_concat_bn(self.fc_img_concat(img_embed_concat))
        # text branch
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))
        #
        img_binary = self.fc_binary(img_embed_concat)
        img_cls = self.fc_cls(img_embed_com)
        img_cls = self.logic(img_cls)

        #img_reconst = self.relu(self.fc_img_reconst(img_cls))

        return img_embed_com, text_embed, img_cls, img_binary #, img_reconst


class MLP_Att_Binary_New(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Att_Binary_New, self).__init__()
        self.fc1_img_com = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_com_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_com = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_com_bn = nn.BatchNorm1d(opt.fc2_size)
        # binary
        self.fc_img_spe = nn.Linear(opt.fc2_size, opt.fc2_size)
        self.fc_img_spe_bn = nn.BatchNorm1d(opt.fc2_size)
        self.fc_img_binary = nn.Linear(opt.fc2_size, opt.fc2_size)
        self.fc_img_binary_bn = nn.BatchNorm1d(opt.fc2_size)

        # image concat
        #self.fc_img_concat = nn.Linear(opt.fc2_size*2, opt.fc2_size)
        #self.fc_img_concat_bn = nn.BatchNorm1d(opt.fc2_size)
        # Att
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        self.fc_cls = nn.Linear(opt.fc2_size, nclass)
        self.fc_binary = nn.Linear(opt.fc2_size, opt.attSize)
        self.fc_img_reconst = nn.Linear(opt.fc2_size, opt.resSize)
        #
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res, att):
        img_embed_com = self.fc1_img_com_bn(self.fc1_img_com(res))
        img_embed_com = self.relu(img_embed_com)
        img_embed_com = self.dropout(img_embed_com)
        img_embed_com = self.fc2_img_com_bn(self.fc2_img_com(img_embed_com))
        #img_embed_com = self.relu(img_embed_com)
        #
        img_embed_sep = self.fc_img_spe_bn(self.fc_img_spe(img_embed_com))
        img_embed_binary = self.fc_img_binary_bn(self.fc_img_binary(img_embed_com))
        # fuse two embed
        img_binary = self.fc_binary(img_embed_binary)
        #img_embed_concat = torch.cat((img_embed_sep, img_embed_binary), 1)
        #img_embed_concat = img_embed_concat + img_embed_com
        #img_embed_concat = self.relu(img_embed_concat)
        #img_embed_concat = self.fc_img_concat_bn(self.fc_img_concat(img_embed_concat))
        #
        img_embed_concat = img_embed_sep + img_embed_binary #torch.cat((img_embed_sep, img_embed_binary), 1)
        img_cls = self.fc_cls(img_embed_concat)
        img_cls = self.logic(img_cls)
        img_reconst = self.relu(self.fc_img_reconst(img_embed_concat))
        # text branch
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))

        return img_embed_concat, text_embed, img_cls, img_binary, img_reconst


class MLP_OneWay_Binary(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_OneWay_Binary, self).__init__()
        self.fc1_img_com = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_com_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_com = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_com_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        #self.fc_cls = nn.Linear(opt.fc2_size, nclass)
        self.fc_binary = nn.Linear(opt.fc2_size, opt.attSize)
        self.fc_binary_bn = nn.BatchNorm1d(opt.attSize)
        #
        #self.fc_img_reconst = nn.Linear(nclass, opt.resSize)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res, att):
        img_embed_com = self.fc1_img_com_bn(self.fc1_img_com(res))
        img_embed_com = self.relu(img_embed_com)
        img_embed_com = self.dropout(img_embed_com)
        img_embed_com = self.fc2_img_com_bn(self.fc2_img_com(img_embed_com))
        #
        # fuse two embed
        #img_embed_concat = img_embed_com + img_embed_binary
        # text branch
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))
        #
        #img_cls = self.fc_cls(img_embed_com)
        #img_cls = self.logic(img_cls)
        img_binary = self.fc_binary_bn(self.fc_binary(img_embed_com))
        #img_reconst = self.relu(self.fc_img_reconst(img_cls))

        return img_embed_com, text_embed, img_binary #, img_binary #, img_reconst


class MLP_Rank(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Rank, self).__init__()
        self.fc1_img_com = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_com_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_com = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_com_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        #self.fc_img_reconst = nn.Linear(nclass, opt.resSize)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res, att):
        img_embed_com = self.fc1_img_com_bn(self.fc1_img_com(res))
        img_embed_com = self.relu(img_embed_com)
        img_embed_com = self.dropout(img_embed_com)
        img_embed_com = self.fc2_img_com_bn(self.fc2_img_com(img_embed_com))
        #
        # fuse two embed
        #img_embed_concat = img_embed_com + img_embed_binary
        # text branch
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))

        return img_embed_com, text_embed


class MLP_OneWay(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_OneWay, self).__init__()
        self.fc1_img_com = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_com_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_com = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_com_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        self.fc_cls = nn.Linear(opt.fc2_size, nclass)
        self.fc_binary = nn.Linear(opt.fc2_size, opt.attSize)
        #
        #self.fc_img_reconst = nn.Linear(nclass, opt.resSize)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res, att):
        img_embed_com = self.fc1_img_com_bn(self.fc1_img_com(res))
        img_embed_com = self.relu(img_embed_com)
        img_embed_com = self.dropout(img_embed_com)
        img_embed_com = self.fc2_img_com_bn(self.fc2_img_com(img_embed_com))
        #
        # fuse two embed
        #img_embed_concat = img_embed_com + img_embed_binary
        # text branch
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))
        #
        img_cls = self.fc_cls(img_embed_com)
        img_cls = self.logic(img_cls)
        img_binary = self.fc_binary(img_embed_com)
        #img_reconst = self.relu(self.fc_img_reconst(img_cls))

        return img_embed_com, text_embed, img_cls, img_binary #, img_reconst

class MLP_OneWay_Att_Confuse(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_OneWay_Att_Confuse, self).__init__()
        self.fc1_img_com = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_com_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_com = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_com_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        self.fc1_text_confuse = nn.Linear(opt.nclass_all, opt.fc1_size)
        self.fc1_text_confuse_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text_confuse = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_confuse_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        self.fc_concat = nn.Linear(opt.fc2_size*2, opt.fc2_size)
        self.fc_concat_bn = nn.BatchNorm1d(opt.fc2_size)
        self.fc_cls = nn.Linear(opt.fc2_size, nclass)
        self.fc_binary = nn.Linear(opt.fc2_size+opt.resSize, opt.attSize)
        #self.fc_binary_bn = nn.BatchNorm1d(opt.attSize)
        #
        self.fc_img_reconst = nn.Linear(opt.fc2_size, opt.resSize)
        #
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res, att, att_confuse):
        img_embed_com = self.fc1_img_com_bn(self.fc1_img_com(res))
        img_embed_com = self.relu(img_embed_com)
        img_embed_com = self.dropout(img_embed_com)
        img_embed_com = self.fc2_img_com_bn(self.fc2_img_com(img_embed_com))
        ## classify
        img_reconst = self.relu(self.fc_img_reconst(img_embed_com))
        img_binary_input = torch.cat((img_embed_com, img_reconst), 1)
        img_binary_input = self.relu(img_binary_input)
        img_binary = self.fc_binary(img_binary_input)
        img_cls = self.fc_cls(img_embed_com)
        img_cls = self.logic(img_cls)
        # text branch
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))
        # text branch
        text_confuse_embed = self.fc1_text_confuse_bn(self.fc1_text_confuse(att_confuse))
        text_confuse_embed = self.relu(text_confuse_embed)
        text_confuse_embed = self.dropout(text_confuse_embed)
        text_confuse_embed = self.fc2_text_confuse_bn(self.fc2_text_confuse(text_confuse_embed))
        # att concat
        # concat
        att_concat = torch.cat((text_embed, text_confuse_embed), 1)
        att_concat = self.relu(att_concat)
        att_concat = self.fc_concat_bn(self.fc_concat(att_concat))

        return img_embed_com, att_concat, img_cls, img_binary, img_reconst




class MLP_Text2Image(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Text2Image, self).__init__()
        # image common branch
        self.fc1_img_com = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_com_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_com = nn.Linear(opt.fc1_size, opt.resSize)
        self.fc2_img_com_bn = nn.BatchNorm1d(opt.resSize)
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.resSize)
        self.fc2_text_bn = nn.BatchNorm1d(opt.resSize)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res, att):
        # image branch
        img_embed_com = self.fc1_img_com_bn(self.fc1_img_com(res))
        img_embed_com = self.relu(img_embed_com)
        img_embed_com = self.dropout(img_embed_com)
        img_embed_com = self.fc2_img_com_bn(self.fc2_img_com(img_embed_com))
        #img_cls = self.fc3_img_com(img_embed_com)
        #img_cls = self.logic(img_cls)
        # text branch
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))

        return img_embed_com, text_embed #, img_cls


# class MLP_Image2Text(nn.Module):
#     def __init__(self, opt, nclass):
#         super(MLP_Image2Text, self).__init__()
#         #self.fc1_text = nn.Linear(opt.resSize, nclass)
#         self.fc1_img = nn.Linear(opt.resSize, opt.fc1_size)
#         self.fc1_img_bn = nn.BatchNorm1d(opt.fc1_size)
#         self.fc2_img = nn.Linear(opt.fc1_size, opt.fc2_size)
#         self.fc2_img_bn = nn.BatchNorm1d(opt.fc2_size)
#         self.fc3_img = nn.Linear(opt.fc2_size, opt.attSize)
#         self.fc3_img_bn = nn.BatchNorm1d(opt.attSize)
#         self.dropout = nn.Dropout(p=0.5)
#         self.dropout_2 = nn.Dropout(p=0.2)
#         self.lrelu = nn.LeakyReLU(0.2, True)
#         self.relu = nn.ReLU(True)
#         self.logic = nn.LogSoftmax(dim=1)
#
#     def forward(self, res):
#         # image branch
#         #img_cls = self.fc1_img(res)
#         #img_cls = self.logic(img_cls)
#         # text branch
#         img_embed = self.fc1_img_bn(self.fc1_img(res))
#         img_embed = self.relu(img_embed)
#         img_embed = self.dropout(img_embed)
#         img_embed = self.fc2_img_bn(self.fc2_img(img_embed))
#         img_embed = self.relu(img_embed)
#         img_embed = self.fc3_img_bn(self.fc3_img(img_embed))
#
#         return img_embed #, img_cls

class MLP_Image2Text(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Image2Text, self).__init__()
        #self.fc1_text = nn.Linear(opt.resSize, nclass)
        self.fc1_img = nn.Linear(opt.resSize, opt.attSize)
        self.fc1_img_bn = nn.BatchNorm1d(opt.attSize)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res):
        # image branch
        #img_cls = self.fc1_img(res)
        #img_cls = self.logic(img_cls)
        # text branch
        img_embed = self.fc1_img_bn(self.fc1_img(res))

        return img_embed #, img_cls



class MLP_ImageEmbed(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_ImageEmbed, self).__init__()
        self.fc1_img = nn.Linear(opt.resSize, opt.attSize)
        self.fc1_img_bn = nn.BatchNorm1d(opt.attSize)
        self.fc2_img = nn.Linear(opt.attSize, nclass)
        self.fc2_img_bn = nn.BatchNorm1d(nclass)
        self.fc3_img = nn.Linear(nclass, opt.resSize)
        self.fc3_img_bn = nn.BatchNorm1d(opt.resSize)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res):
        # image branch
        fc1_img_bn = self.fc1_img_bn(self.fc1_img(res))
        fc1_img_relu = self.relu(fc1_img_bn)
        fc1_img_relu = self.dropout(fc1_img_relu)
        fc2_img_bn = self.fc2_img_bn(self.fc2_img(fc1_img_relu))
        fc2_img_relu = self.relu(fc2_img_bn)
        fc3_img_bn = self.fc3_img_bn(self.fc3_img(fc2_img_relu))
        fc3_img_relu = self.relu(fc3_img_bn)
        img_cls = self.logic(fc2_img_bn)

        return fc1_img_bn, img_cls, fc3_img_relu


class MLP_ImgClassifier(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_ImgClassifier, self).__init__()
        self.fc1_img = nn.Linear(opt.resSize, opt.attSize)
        self.fc2_img_spe = nn.Linear(opt.resSize, 50)
        self.fc_img_reconst = nn.Linear(nclass+50, opt.resSize)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res):
        # image branch
        img_com = self.fc1_img_com(res)  ## how about share fc3??
        img_spe = self.fc1_img_spe(res)
        img_embed_concat = torch.cat((img_com, img_spe), 1)
        #img_embed_relu = self.relu(img_com)
        img_reconst = self.relu(self.fc_img_reconst(img_embed_concat))
        img_cls = self.logic(img_com)
        return img_com, img_spe, img_cls, img_reconst


class MLP_Img_TwoWay(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Img_TwoWay, self).__init__()
        # image common branch
        self.fc1_img_com = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_com_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_com = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_com_bn = nn.BatchNorm1d(opt.fc2_size)
        self.fc3_img_com = nn.Linear(opt.fc2_size, nclass)
        # image private branch
        self.fc1_img_spe = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_spe_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_spe = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_spe_bn = nn.BatchNorm1d(opt.fc2_size)
        ##
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        # reconstruct branch
        self.fc_img_text_reconst = nn.Linear(opt.fc2_size*2, opt.resSize)
        self.fc_img_reconst = nn.Linear(opt.fc2_size*2, opt.resSize)
        # others
        self.dropout = nn.Dropout(p=0.5)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, res, att):
        # image common branch
        img_embed_com = self.fc1_img_com_bn(self.fc1_img_com(res))
        img_embed_com = self.relu(img_embed_com)
        img_embed_com = self.dropout(img_embed_com)
        img_embed_com = self.fc2_img_com_bn(self.fc2_img_com(img_embed_com))
        img_cls = self.fc3_img_com(img_embed_com)  ## how about share fc3??
        img_cls = self.logic(img_cls)
        # image private branch
        img_embed_spe = self.fc1_img_spe_bn(self.fc1_img_spe(res))
        img_embed_spe = self.relu(img_embed_spe)
        img_embed_spe = self.dropout(img_embed_spe)
        img_embed_spe = self.fc2_img_spe_bn(self.fc2_img_spe(img_embed_spe))
        # att agg
        # text branch
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))
        # reconsturct text feature
        img_embed_concat = torch.cat((img_embed_com, img_embed_spe), 1)
        img_embed_concat = self.relu(img_embed_concat)
        img_reconst = self.relu(self.fc_img_reconst(img_embed_concat))
        #
        img_text_concat = torch.cat((text_embed, img_embed_spe), 1)
        img_text_concat = self.relu(img_text_concat)
        img_text_reconst = self.relu(self.fc_img_text_reconst(img_text_concat)) # share weights with img_reconst

        return img_embed_com, img_embed_spe, img_cls, text_embed, img_reconst, img_text_reconst


class MLP_Att_TwoWay_Avg(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Att_TwoWay_Avg, self).__init__()
        # img
        self.fc1_img = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_bn = nn.BatchNorm1d(opt.fc2_size)
        self.fc3_img = nn.Linear(opt.fc2_size, nclass)
        # att
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        # att agg
        self.fc1_agg = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_agg_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_agg = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_agg_bn = nn.BatchNorm1d(opt.fc2_size)
        # concat
        self.fc3_concat = nn.Linear(opt.fc2_size*2, opt.fc2_size)
        self.fc3_concat_bn = nn.BatchNorm1d(opt.fc2_size)
        # others
        self.dropout = nn.Dropout(p=0.5)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, opt, res, att, att_agg):
        # image branch
        img_embed = self.fc1_img_bn(self.fc1_img(res))
        img_embed = self.relu(img_embed)
        img_embed = self.dropout(img_embed)
        img_embed = self.fc2_img_bn(self.fc2_img(img_embed))
        img_cls = self.fc3_img(img_embed)  ## how about share fc3??
        img_cls = self.logic(img_cls)
        # att
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))
        #text_embed = self.relu(text_embed)
        # att agg
        att_agg = att_agg.view(-1, opt.nclass_all, opt.attSize, 1)
        att_mean = att_agg.mean(1)
        att_mean = att_mean.view(-1, opt.attSize)
        text_embed_agg = self.fc1_agg_bn(self.fc1_agg(att_mean))
        text_embed_agg = self.relu(text_embed_agg)
        text_embed_agg = self.dropout(text_embed_agg)
        text_embed_agg = self.fc2_agg_bn(self.fc2_agg(text_embed_agg))
        # concat
        att_concat = torch.cat((text_embed, text_embed_agg), 1)
        att_concat = self.relu(att_concat)
        att_concat_embed = self.fc3_concat_bn(self.fc3_concat(att_concat))

        return img_embed, img_cls, text_embed, att_concat_embed


class MLP_Att_TwoWay_Conv(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Att_TwoWay_Conv, self).__init__()
        # img
        self.fc1_img = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_bn = nn.BatchNorm1d(opt.fc2_size)
        self.fc3_img = nn.Linear(opt.fc2_size, nclass)
        # att
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        # att agg
        self.conv1 = nn.Conv2d(opt.nclass_all, 1, kernel_size=1, stride=1, padding=0)
        self.fc1_agg = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_agg_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_agg = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_agg_bn = nn.BatchNorm1d(opt.fc2_size)
        # concat
        self.fc3_concat = nn.Linear(opt.fc2_size*2, opt.fc2_size)
        self.fc3_concat_bn = nn.BatchNorm1d(opt.fc2_size)
        #
        self.dropout = nn.Dropout(p=0.5)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.fill_(1.0)
        self.conv1.bias.data.fill_(0.0)

    def forward(self, opt, res, att, att_agg):
        # image branch
        img_embed = self.fc1_img_bn(self.fc1_img(res))
        img_embed = self.relu(img_embed)
        img_embed = self.dropout(img_embed)
        img_embed = self.fc2_img_bn(self.fc2_img(img_embed))
        img_cls = self.fc3_img(img_embed)  ## how about share fc3??
        img_cls = self.logic(img_cls)
        # att
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))
        #text_embed = self.relu(text_embed)
        # att agg
        att_agg = att_agg.view(-1, opt.nclass_all, opt.attSize, 1)
        text_conv1 = self.relu(self.conv1(att_agg))
        text_conv1 = text_conv1.view(-1, opt.attSize)
        #text_conv1 = self.conv1_bn(text_conv1)
        text_embed_agg = self.fc1_agg_bn(self.fc1_agg(text_conv1))
        text_embed_agg = self.relu(text_embed_agg)
        text_embed_agg = self.dropout(text_embed_agg)
        text_embed_agg = self.fc2_agg_bn(self.fc2_agg(text_embed_agg))
        # concat
        att_concat_embed = torch.cat((text_embed, text_embed_agg), 1)
        att_concat_embed = self.relu(att_concat_embed)
        att_concat_embed = self.fc3_concat_bn(self.fc3_concat(att_concat_embed))

        return img_embed, att_concat_embed, img_cls


class MLP_Img_Att_TwoWay_Avg(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Img_Att_TwoWay_Avg, self).__init__()
        # image common branch
        self.fc1_img_com = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_com_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_com = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_com_bn = nn.BatchNorm1d(opt.fc2_size)
        self.fc3_img_com = nn.Linear(opt.fc2_size, nclass)
        # image private branch
        self.fc1_img_spe = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_spe_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_spe = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_spe_bn = nn.BatchNorm1d(opt.fc2_size)
        # att
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        # att agg
        self.fc1_agg = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_agg_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_agg = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_agg_bn = nn.BatchNorm1d(opt.fc2_size)
        # concat
        self.fc_att_concat = nn.Linear(opt.fc2_size*2, opt.fc2_size)
        self.fc_att_concat_bn = nn.BatchNorm1d(opt.fc2_size)
        # reconstruct branch
        self.fc_img_text_reconst = nn.Linear(opt.fc2_size*2, opt.resSize)
        self.fc_img_reconst = nn.Linear(opt.fc2_size*2, opt.resSize)
        self.fc_img_text_agg_reconst = nn.Linear(opt.fc2_size*2, opt.resSize)
        #
        self.dropout = nn.Dropout(p=0.5)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, opt, res, att, att_agg):
        # image common branch
        img_embed_com = self.fc1_img_com_bn(self.fc1_img_com(res))
        img_embed_com = self.relu(img_embed_com)
        img_embed_com = self.dropout(img_embed_com)
        img_embed_com = self.fc2_img_com_bn(self.fc2_img_com(img_embed_com))
        img_cls = self.fc3_img_com(img_embed_com)  ## how about share fc3??
        img_cls = self.logic(img_cls)
        # image private branch
        img_embed_spe = self.fc1_img_spe_bn(self.fc1_img_spe(res))
        img_embed_spe = self.relu(img_embed_spe)
        img_embed_spe = self.dropout(img_embed_spe)
        img_embed_spe = self.fc2_img_spe_bn(self.fc2_img_spe(img_embed_spe))
        # att
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))
        # att agg
        att_agg = att_agg.view(-1, opt.nclass_all, opt.attSize, 1)
        att_mean = att_agg.mean(1)
        att_mean = att_mean.view(-1, opt.attSize)
        text_embed_agg = self.fc1_agg_bn(self.fc1_agg(att_mean))
        text_embed_agg = self.relu(text_embed_agg)
        text_embed_agg = self.dropout(text_embed_agg)
        text_embed_agg = self.fc2_agg_bn(self.fc2_agg(text_embed_agg))
        # concat
        #att_concat_embed = text_embed + text_embed_agg
        att_concat = torch.cat((text_embed, text_embed_agg), 1)
        att_concat = self.relu(att_concat)
        att_concat_embed = self.fc_att_concat_bn(self.fc_att_concat(att_concat))
        # reconsturct text feature
        img_embed_concat = torch.cat((img_embed_com, img_embed_spe), 1)
        img_embed_concat = self.relu(img_embed_concat)
        img_reconst = self.relu(self.fc_img_reconst(img_embed_concat))
        #
        img_text_concat = torch.cat((text_embed, img_embed_spe), 1)
        img_text_concat = self.relu(img_text_concat)
        img_text_reconst = self.relu(self.fc_img_text_reconst(img_text_concat)) ## share weights
        #
        img_text_agg_concat = torch.cat((att_concat_embed, img_embed_spe), 1)
        img_text_agg_concat = self.relu(img_text_agg_concat)
        img_text_agg_reconst = self.relu(self.fc_img_text_agg_reconst(img_text_agg_concat)) ## share weights

        return img_embed_com, img_embed_spe, img_cls, text_embed, att_concat_embed, img_reconst, img_text_reconst, img_text_agg_reconst


class MLP_Img_Att_TwoWay_Conv(nn.Module):
    def __init__(self, opt, nclass):
        super(MLP_Img_Att_TwoWay_Conv, self).__init__()
        # image common branch
        self.fc1_img_com = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_com_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_com = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_com_bn = nn.BatchNorm1d(opt.fc2_size)
        self.fc3_img_com = nn.Linear(opt.fc2_size, nclass)
        # image private branch
        self.fc1_img_spe = nn.Linear(opt.resSize, opt.fc1_size)
        self.fc1_img_spe_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_img_spe = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_img_spe_bn = nn.BatchNorm1d(opt.fc2_size)
        # att
        self.fc1_text = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_text_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_text = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_text_bn = nn.BatchNorm1d(opt.fc2_size)
        # att agg
        self.conv1 = nn.Conv2d(opt.nclass_all, 1, kernel_size=1, stride=1, padding=0)
        self.fc1_agg = nn.Linear(opt.attSize, opt.fc1_size)
        self.fc1_agg_bn = nn.BatchNorm1d(opt.fc1_size)
        self.fc2_agg = nn.Linear(opt.fc1_size, opt.fc2_size)
        self.fc2_agg_bn = nn.BatchNorm1d(opt.fc2_size)
        # concat
        self.fc_att_concat = nn.Linear(opt.fc2_size*2, opt.fc2_size)
        self.fc_att_concat_bn = nn.BatchNorm1d(opt.fc2_size)
        # reconstruct branch
        #self.fc_img_text_reconst = nn.Linear(opt.fc2_size*2, opt.resSize)
        self.fc_img_reconst = nn.Linear(opt.fc2_size*2, opt.resSize)
        self.fc_img_text_agg_reconst = nn.Linear(opt.fc2_size*2, opt.resSize)
        #
        self.dropout = nn.Dropout(p=0.5)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.fill_(0.5)
        self.conv1.bias.data.fill_(0.0)

    def forward(self, opt, res, att, att_agg):
        # image common branch
        img_embed_com = self.fc1_img_com_bn(self.fc1_img_com(res))
        img_embed_com = self.relu(img_embed_com)
        img_embed_com = self.dropout(img_embed_com)
        img_embed_com = self.fc2_img_com_bn(self.fc2_img_com(img_embed_com))
        img_cls = self.fc3_img_com(img_embed_com)  ## how about share fc3??
        img_cls = self.logic(img_cls)
        # image private branch
        img_embed_spe = self.fc1_img_spe_bn(self.fc1_img_spe(res))
        img_embed_spe = self.relu(img_embed_spe)
        img_embed_spe = self.dropout(img_embed_spe)
        img_embed_spe = self.fc2_img_spe_bn(self.fc2_img_spe(img_embed_spe))
        # att
        text_embed = self.fc1_text_bn(self.fc1_text(att))
        text_embed = self.relu(text_embed)
        text_embed = self.dropout(text_embed)
        text_embed = self.fc2_text_bn(self.fc2_text(text_embed))
        # att agg
        att_agg = att_agg.view(-1, opt.nclass_all, opt.attSize, 1)
        text_conv1 = self.relu(self.conv1(att_agg))
        text_conv1 = text_conv1.view(-1, opt.attSize)
        text_embed_agg = self.fc1_agg_bn(self.fc1_agg(text_conv1))
        text_embed_agg = self.relu(text_embed_agg)
        text_embed_agg = self.dropout(text_embed_agg)
        text_embed_agg = self.fc2_agg_bn(self.fc2_agg(text_embed_agg))
        # concat
        #att_concat_embed = text_embed + text_embed_agg
        att_concat = torch.cat((text_embed, text_embed_agg), 1)
        att_concat = self.relu(att_concat)
        att_concat_embed = self.fc_att_concat_bn(self.fc_att_concat(att_concat))
        # reconsturct text feature
        img_embed_concat = torch.cat((img_embed_com, img_embed_spe), 1)
        img_embed_concat = self.relu(img_embed_concat)
        img_reconst = self.relu(self.fc_img_reconst(img_embed_concat))
        #
        # img_text_concat = torch.cat((text_embed, img_embed_spe), 1)
        # img_text_concat = self.relu(img_text_concat)
        # img_text_reconst = self.relu(self.fc_img_text_reconst(img_text_concat))
        #
        img_text_agg_concat = torch.cat((att_concat_embed, img_embed_spe), 1)
        img_text_agg_concat = self.relu(img_text_agg_concat)
        img_text_agg_reconst = self.relu(self.fc_img_text_agg_reconst(img_text_agg_concat))

        return img_embed_com, img_embed_spe, img_cls, att_concat_embed, img_reconst, img_text_agg_reconst



###########################################################

class MLP_RNN_OneLSTM_Res(nn.Module):
    def __init__(self, opt):  # (embed_size, hidden_size, vocab_size, num_layers, max_seq_length=50):
        """Set the hyper-parameters and build the layers."""
        super(MLP_RNN_OneLSTM_Res, self).__init__()
        self.linear1 = nn.Linear(opt.resSize, opt.embed_size)  # image feature is fed into the second LSTM
        self.word_embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        ## LSTM setting
        lstm1_input_size = opt.embed_size + opt.hidden_size
        self.lstm1 = nn.LSTM(lstm1_input_size, opt.hidden_size, batch_first=True)
        self.fc_cls = nn.Linear(opt.hidden_size, opt.vocab_size)
        self.max_seg_length = opt.max_seq_length
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        self.linear1.bias.data.fill_(0)
        self.fc_cls.weight.data.uniform_(-0.1, 0.1)
        self.fc_cls.bias.data.fill_(0)

    def forward(self, img_res, wordID, lengths):
        # embed word vectors
        embeddings = self.word_embed(wordID)
        img_res = self.relu(self.linear1(img_res))
        img_res = img_res.unsqueeze(1)
        img_res = img_res.expand(-1, self.max_seg_length, -1)
        lstm1_input = torch.cat((img_res, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm1_input, lengths, batch_first=True)
        hiddens, _ = self.lstm1(packed_embed)
        lstm1_hiddens, _ = self.lstm1(packed_embed)
        lstm1_hiddens = self.dropout(lstm1_hiddens[0])
        outputs = self.fc_cls(lstm1_hiddens)

        return outputs

    def generate_sentence(self, img_res, start_word, end_word, states=(None,None, None), max_sampling_length=30, sample=False):
        sampled_ids = []
        embedded_word = self.word_embed(start_word) # start_word is [1, 1] ?
        img_res = self.relu(self.linear1(img_res))
        img_res = img_res.unsqueeze(1)
        embedded_word = embedded_word.expand(img_res.size(0), -1, -1)
        lstm1_states, lstm2_states, lstm3_states = states

        end_word = end_word.squeeze().expand(img_res.size(0))
        reached_end = torch.zeros_like(end_word.data).byte()
        if sample:
            log_probabilities = []
            lengths = torch.zeros_like(reached_end).long()

        i = 0
        while not reached_end.all() and i < max_sampling_length:
            # LSTM 1
            lstm1_input = torch.cat((img_res, embedded_word), 2)  # start with <start_word>
            lstm1_hiddens, lstm1_states = self.lstm1(lstm1_input, lstm1_states)
            outputs = self.fc_cls(lstm1_hiddens.squeeze(1))

            if sample:
                predicted, log_p = self.sample(outputs)
                active_batches = (~reached_end)
                log_p *= active_batches.float().to(log_p.device)
                log_probabilities.append(log_p.unsqueeze(1))
                lengths += active_batches.long()
            else:
                predicted = outputs.max(1)[1]
            reached_end = reached_end | predicted.eq(end_word).data
            sampled_ids.append(predicted.unsqueeze(1))
            embedded_word = self.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()
        if sample:
            log_probabilities = torch.cat(log_probabilities, 1).squeeze()
            return sampled_ids, log_probabilities, lengths

        return sampled_ids


class MLP_RNN_OneLSTM_Embed(nn.Module):
    def __init__(self, opt):  # (embed_size, hidden_size, vocab_size, num_layers, max_seq_length=50):
        """Set the hyper-parameters and build the layers."""
        super(MLP_RNN_OneLSTM_Embed, self).__init__()
        self.word_embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        ## LSTM setting
        lstm1_input_size = opt.fc2_size + opt.hidden_size
        self.lstm1 = nn.LSTM(lstm1_input_size, opt.hidden_size, batch_first=True)
        self.fc_cls = nn.Linear(opt.hidden_size, opt.vocab_size)
        self.max_seg_length = opt.max_seq_length
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.fc_cls.weight.data.uniform_(-0.1, 0.1)
        self.fc_cls.bias.data.fill_(0)

    def forward(self, img_embed, wordID, lengths):
        # embed word vectors
        embeddings = self.word_embed(wordID)
        # LSTM 1
        img_embed = img_embed.unsqueeze(1)
        img_embed = img_embed.expand(-1, self.max_seg_length, -1)
        lstm1_input = torch.cat((img_embed, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm1_input, lengths, batch_first=True)
        lstm1_hiddens, _ = self.lstm1(packed_embed)
        lstm1_hiddens = self.dropout(lstm1_hiddens[0])
        outputs = self.fc_cls(lstm1_hiddens)

        return outputs

    def generate_sentence(self, img_embed, start_word, end_word, states=(None,None,None), max_sampling_length=30, sample=False):
        sampled_ids = []
        embedded_word = self.word_embed(start_word) # start_word is [1, 1] ?
        img_embed = img_embed.unsqueeze(1)
        embedded_word = embedded_word.expand(img_embed.size(0), -1, -1)
        lstm1_states, lstm2_states, lstm3_states = states
        end_word = end_word.squeeze().expand(img_embed.size(0))
        reached_end = torch.zeros_like(end_word.data).byte()
        if sample:
            log_probabilities = []
            lengths = torch.zeros_like(reached_end).long()

        i = 0
        while not reached_end.all() and i < max_sampling_length:
            # LSTM 1
            lstm1_input = torch.cat((img_embed, embedded_word), 2)  # start with <start_word>
            lstm1_hiddens, lstm1_states = self.lstm1(lstm1_input, lstm1_states)
            outputs = self.fc_cls(lstm1_hiddens.squeeze(1))

            if sample:
                predicted, log_p = self.sample(outputs)
                active_batches = (~reached_end)
                log_p *= active_batches.float().to(log_p.device)
                log_probabilities.append(log_p.unsqueeze(1))
                lengths += active_batches.long()
            else:
                predicted = outputs.max(1)[1]
            reached_end = reached_end | predicted.eq(end_word).data
            sampled_ids.append(predicted.unsqueeze(1))
            embedded_word = self.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()
        if sample:
            log_probabilities = torch.cat(log_probabilities, 1).squeeze()
            return sampled_ids, log_probabilities, lengths

        return sampled_ids


class MLP_RNN_OneLSTM_Att(nn.Module):
    def __init__(self, opt):  # (embed_size, hidden_size, vocab_size, num_layers, max_seq_length=50):
        """Set the hyper-parameters and build the layers."""
        super(MLP_RNN_OneLSTM_Att, self).__init__()
        self.word_embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        ## LSTM setting
        lstm1_input_size = opt.attSize + opt.hidden_size
        self.lstm1 = nn.LSTM(lstm1_input_size, opt.hidden_size, batch_first=True)
        self.fc_cls = nn.Linear(opt.hidden_size, opt.vocab_size)
        self.max_seg_length = opt.max_seq_length
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.fc_cls.weight.data.uniform_(-0.1, 0.1)
        self.fc_cls.bias.data.fill_(0)

    def forward(self, img_binary, wordID, lengths):
        # embed word vectors
        embeddings = self.word_embed(wordID)
        # LSTM 1
        img_binary = img_binary.unsqueeze(1)
        img_binary = img_binary.expand(-1, self.max_seg_length, -1)
        lstm1_input = torch.cat((img_binary, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm1_input, lengths, batch_first=True)
        lstm1_hiddens, _ = self.lstm1(packed_embed)
        lstm1_hiddens = self.dropout(lstm1_hiddens[0])
        outputs = self.fc_cls(lstm1_hiddens)

        return outputs

    def generate_sentence(self, img_binary, start_word, end_word, states=(None,None,None), max_sampling_length=30, sample=False):
        sampled_ids = []
        embedded_word = self.word_embed(start_word) # start_word is [1, 1] ?
        #img_res = self.relu(self.linear1_1(img_res))
        img_binary = img_binary.unsqueeze(1)
        embedded_word = embedded_word.expand(img_binary.size(0), -1, -1)
        lstm1_states, lstm2_states, lstm3_states = states
        end_word = end_word.squeeze().expand(img_binary.size(0))
        reached_end = torch.zeros_like(end_word.data).byte()
        if sample:
            log_probabilities = []
            lengths = torch.zeros_like(reached_end).long()

        i = 0
        while not reached_end.all() and i < max_sampling_length:
            # LSTM 1
            lstm1_input = torch.cat((img_binary, embedded_word), 2)  # start with <start_word>
            lstm1_hiddens, lstm1_states = self.lstm1(lstm1_input, lstm1_states)
            outputs = self.fc_cls(lstm1_hiddens.squeeze(1))

            if sample:
                predicted, log_p = self.sample(outputs)
                active_batches = (~reached_end)
                log_p *= active_batches.float().to(log_p.device)
                log_probabilities.append(log_p.unsqueeze(1))
                lengths += active_batches.long()
            else:
                predicted = outputs.max(1)[1]
            reached_end = reached_end | predicted.eq(end_word).data
            sampled_ids.append(predicted.unsqueeze(1))
            embedded_word = self.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()
        if sample:
            log_probabilities = torch.cat(log_probabilities, 1).squeeze()
            return sampled_ids, log_probabilities, lengths

        return sampled_ids


class MLP_RNN_TwoLSTM_Res_Embed(nn.Module):
    def __init__(self, opt):  # (embed_size, hidden_size, vocab_size, num_layers, max_seq_length=50):
        """Set the hyper-parameters and build the layers."""
        super(MLP_RNN_TwoLSTM_Res_Embed, self).__init__()
        self.linear1 = nn.Linear(opt.resSize, opt.embed_size)  # image feature is fed into the second LSTM
        #self.linear2 = nn.Linear(opt.fc2_size, opt.embed_size)
        self.word_embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        ## LSTM setting
        lstm1_input_size = opt.embed_size + opt.hidden_size
        self.lstm1 = nn.LSTM(lstm1_input_size, opt.hidden_size, batch_first=True)
        lstm2_input_size = opt.fc2_size + opt.hidden_size
        self.lstm2 = nn.LSTM(lstm2_input_size, opt.hidden_size, batch_first=True)
        self.fc_cls = nn.Linear(opt.hidden_size*2, opt.vocab_size)
        self.max_seg_length = opt.max_seq_length
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        self.linear1.bias.data.fill_(0)
        self.fc_cls.weight.data.uniform_(-0.1, 0.1)
        self.fc_cls.bias.data.fill_(0)

    def forward(self, img_res, img_embed, wordID, lengths):
        # embed word vectors
        embeddings = self.word_embed(wordID)
        # LSTM 1
        img_res = self.relu(self.linear1(img_res))
        img_res = img_res.unsqueeze(1)
        img_res = img_res.expand(-1, self.max_seg_length, -1)
        lstm1_input = torch.cat((img_res, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm1_input, lengths, batch_first=True)
        lstm1_hiddens, _ = self.lstm1(packed_embed)
        lstm1_hiddens = self.dropout(lstm1_hiddens[0])

        # LSTM 2
        #img_embed = self.relu(self.linear2(img_embed))
        img_embed = img_embed.unsqueeze(1)
        img_embed = img_embed.expand(-1, self.max_seg_length, -1)
        lstm2_input = torch.cat((img_embed, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm2_input, lengths, batch_first=True)
        lstm2_hiddens, _ = self.lstm2(packed_embed)
        lstm2_hiddens = self.dropout(lstm2_hiddens[0])

        lstm_hiddens = torch.cat((lstm1_hiddens, lstm2_hiddens), 1)
        outputs = self.fc_cls(lstm_hiddens)

        return outputs

    def generate_sentence(self, img_res, img_embed, start_word, end_word, states=(None,None,None), max_sampling_length=30, sample=False):
        sampled_ids = []
        embedded_word = self.word_embed(start_word) # start_word is [1, 1] ?
        img_res = self.relu(self.linear1(img_res))
        img_res = img_res.unsqueeze(1)
        #img_embed = self.relu(self.linear2(img_embed))
        img_embed = img_embed.unsqueeze(1)

        embedded_word = embedded_word.expand(img_res.size(0), -1, -1)
        lstm1_states, lstm2_states, lstm3_states = states
        end_word = end_word.squeeze().expand(img_res.size(0))
        reached_end = torch.zeros_like(end_word.data).byte()
        if sample:
            log_probabilities = []
            lengths = torch.zeros_like(reached_end).long()

        i = 0
        while not reached_end.all() and i < max_sampling_length:
            # LSTM 1
            lstm1_input = torch.cat((img_res, embedded_word), 2)  # start with <start_word>
            lstm1_hiddens, lstm1_states = self.lstm1(lstm1_input, lstm1_states)

            # LSTM 2
            lstm2_input = torch.cat((img_embed, embedded_word), 2)
            lstm2_hiddens, lstm2_states = self.lstm2(lstm2_input, lstm2_states)

            lstm_hiddens = torch.cat((lstm1_hiddens.squeeze(1), lstm2_hiddens.squeeze(1)), 1)
            outputs = self.fc_cls(lstm_hiddens)

            if sample:
                predicted, log_p = self.sample(outputs)
                active_batches = (~reached_end)
                log_p *= active_batches.float().to(log_p.device)
                log_probabilities.append(log_p.unsqueeze(1))
                lengths += active_batches.long()
            else:
                predicted = outputs.max(1)[1]
            reached_end = reached_end | predicted.eq(end_word).data
            sampled_ids.append(predicted.unsqueeze(1))
            embedded_word = self.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()
        if sample:
            log_probabilities = torch.cat(log_probabilities, 1).squeeze()
            return sampled_ids, log_probabilities, lengths

        return sampled_ids


class MLP_RNN_TwoLSTM_Embed_Att(nn.Module):
    def __init__(self, opt):  # (embed_size, hidden_size, vocab_size, num_layers, max_seq_length=50):
        """Set the hyper-parameters and build the layers."""
        super(MLP_RNN_TwoLSTM_Embed_Att, self).__init__()
        #self.linear1 = nn.Linear(opt.fc2_size, opt.embed_size)  # image feature is fed into the second LSTM
        #self.linear2 = nn.Linear(opt.attSize, opt.embed_size)
        self.word_embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        ## LSTM setting
        lstm1_input_size = opt.fc2_size + opt.hidden_size
        self.lstm1 = nn.LSTM(lstm1_input_size, opt.hidden_size, batch_first=True)
        lstm2_input_size = opt.attSize + opt.hidden_size
        self.lstm2 = nn.LSTM(lstm2_input_size, opt.hidden_size, batch_first=True)

        self.fc_cls = nn.Linear(opt.hidden_size*2, opt.vocab_size)
        self.max_seg_length = opt.max_seq_length
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        #self.linear1.weight.data.uniform_(-0.1, 0.1)
        #self.linear1.bias.data.fill_(0)
        #self.linear2.weight.data.uniform_(-0.1, 0.1)
        #self.linear2.bias.data.fill_(0)
        self.fc_cls.weight.data.uniform_(-0.1, 0.1)
        self.fc_cls.bias.data.fill_(0)

    def forward(self, img_embed, img_binary, wordID, lengths):
        # embed word vectors
        embeddings = self.word_embed(wordID)
        # LSTM 1
        #img_embed = self.relu(self.linear1(img_embed))
        img_embed = img_embed.unsqueeze(1)
        img_embed = img_embed.expand(-1, self.max_seg_length, -1)
        lstm1_input = torch.cat((img_embed, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm1_input, lengths, batch_first=True)
        lstm1_hiddens, _ = self.lstm1(packed_embed)
        lstm1_hiddens = self.dropout(lstm1_hiddens[0])

        # LSTM 2
        #img_binary = self.relu(self.linear2(img_binary))
        img_binary = img_binary.unsqueeze(1)
        img_binary = img_binary.expand(-1, self.max_seg_length, -1)
        lstm2_input = torch.cat((img_binary, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm2_input, lengths, batch_first=True)
        lstm2_hiddens, _ = self.lstm2(packed_embed)
        lstm2_hiddens = self.dropout(lstm2_hiddens[0])

        lstm_hiddens = torch.cat((lstm1_hiddens, lstm2_hiddens), 1)
        outputs = self.fc_cls(lstm_hiddens)

        return outputs

    def generate_sentence(self, img_embed, img_binary, start_word, end_word, states=(None,None,None), max_sampling_length=30, sample=False):
        sampled_ids = []
        embedded_word = self.word_embed(start_word) # start_word is [1, 1] ?
        #img_embed = self.relu(self.linear1(img_embed))
        img_embed = img_embed.unsqueeze(1)
        #img_binary = self.relu(self.linear2(img_binary))
        img_binary = img_binary.unsqueeze(1)

        embedded_word = embedded_word.expand(img_embed.size(0), -1, -1)
        lstm1_states, lstm2_states, lstm3_states = states
        end_word = end_word.squeeze().expand(img_embed.size(0))
        reached_end = torch.zeros_like(end_word.data).byte()
        if sample:
            log_probabilities = []
            lengths = torch.zeros_like(reached_end).long()

        i = 0
        while not reached_end.all() and i < max_sampling_length:
            # LSTM 1
            lstm1_input = torch.cat((img_embed, embedded_word), 2)  # start with <start_word>
            lstm1_hiddens, lstm1_states = self.lstm1(lstm1_input, lstm1_states)

            # LSTM 2
            lstm2_input = torch.cat((img_binary, embedded_word), 2)
            lstm2_hiddens, lstm2_states = self.lstm2(lstm2_input, lstm2_states)

            lstm_hiddens = torch.cat((lstm1_hiddens.squeeze(1), lstm2_hiddens.squeeze(1)), 1)
            outputs = self.fc_cls(lstm_hiddens)

            if sample:
                predicted, log_p = self.sample(outputs)
                active_batches = (~reached_end)
                log_p *= active_batches.float().to(log_p.device)
                log_probabilities.append(log_p.unsqueeze(1))
                lengths += active_batches.long()
            else:
                predicted = outputs.max(1)[1]
            reached_end = reached_end | predicted.eq(end_word).data
            sampled_ids.append(predicted.unsqueeze(1))
            embedded_word = self.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()
        if sample:
            log_probabilities = torch.cat(log_probabilities, 1).squeeze()
            return sampled_ids, log_probabilities, lengths

        return sampled_ids


class MLP_RNN_TwoLSTM_Res_Att(nn.Module):
    def __init__(self, opt):  # (embed_size, hidden_size, vocab_size, num_layers, max_seq_length=50):
        """Set the hyper-parameters and build the layers."""
        super(MLP_RNN_TwoLSTM_Res_Att, self).__init__()
        self.linear1 = nn.Linear(opt.resSize, opt.embed_size)  # image feature is fed into the second LSTM
        self.word_embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        ## LSTM setting
        lstm1_input_size = opt.embed_size + opt.hidden_size
        self.lstm1 = nn.LSTM(lstm1_input_size, opt.hidden_size, batch_first=True)
        lstm2_input_size = opt.attSize + opt.hidden_size
        self.lstm2 = nn.LSTM(lstm2_input_size, opt.hidden_size, batch_first=True)
        self.fc_cls = nn.Linear(opt.hidden_size*2, opt.vocab_size)
        self.max_seg_length = opt.max_seq_length
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        self.linear1.bias.data.fill_(0)
        self.fc_cls.weight.data.uniform_(-0.1, 0.1)
        self.fc_cls.bias.data.fill_(0)

    def forward(self, img_res, img_binary, wordID, lengths):
        # embed word vectors
        embeddings = self.word_embed(wordID)
        # LSTM 1
        img_res = self.relu(self.linear1(img_res))
        img_res = img_res.unsqueeze(1)
        img_res = img_res.expand(-1, self.max_seg_length, -1)
        lstm1_input = torch.cat((img_res, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm1_input, lengths, batch_first=True)
        lstm1_hiddens, _ = self.lstm1(packed_embed)
        lstm1_hiddens = self.dropout(lstm1_hiddens[0])
        # LSTM 2
        img_binary = img_binary.unsqueeze(1)
        img_binary = img_binary.expand(-1, self.max_seg_length, -1)
        lstm2_input = torch.cat((img_binary, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm2_input, lengths, batch_first=True)
        lstm2_hiddens, _ = self.lstm2(packed_embed)
        lstm2_hiddens = self.dropout(lstm2_hiddens[0])

        lstm_hiddens = torch.cat((lstm1_hiddens, lstm2_hiddens), 1)
        outputs = self.fc_cls(lstm_hiddens)

        return outputs

    def generate_sentence(self, img_res, img_binary, start_word, end_word, states=(None,None,None), max_sampling_length=30, sample=False):
        sampled_ids = []
        embedded_word = self.word_embed(start_word) # start_word is [1, 1] ?
        img_res = self.relu(self.linear1(img_res))
        img_res = img_res.unsqueeze(1)
        img_binary = img_binary.unsqueeze(1)

        embedded_word = embedded_word.expand(img_res.size(0), -1, -1)
        lstm1_states, lstm2_states, lstm3_states = states
        end_word = end_word.squeeze().expand(img_res.size(0))
        reached_end = torch.zeros_like(end_word.data).byte()
        if sample:
            log_probabilities = []
            lengths = torch.zeros_like(reached_end).long()

        i = 0
        while not reached_end.all() and i < max_sampling_length:
            # LSTM 1
            lstm1_input = torch.cat((img_res, embedded_word), 2)  # start with <start_word>
            lstm1_hiddens, lstm1_states = self.lstm1(lstm1_input, lstm1_states)

            # LSTM 2
            lstm2_input = torch.cat((img_binary, embedded_word), 2)
            lstm2_hiddens, lstm2_states = self.lstm2(lstm2_input, lstm2_states)

            lstm_hiddens = torch.cat((lstm1_hiddens.squeeze(1), lstm2_hiddens.squeeze(1)), 1)
            outputs = self.fc_cls(lstm_hiddens)

            if sample:
                predicted, log_p = self.sample(outputs)
                active_batches = (~reached_end)
                log_p *= active_batches.float().to(log_p.device)
                log_probabilities.append(log_p.unsqueeze(1))
                lengths += active_batches.long()
            else:
                predicted = outputs.max(1)[1]
            reached_end = reached_end | predicted.eq(end_word).data
            sampled_ids.append(predicted.unsqueeze(1))
            embedded_word = self.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()
        if sample:
            log_probabilities = torch.cat(log_probabilities, 1).squeeze()
            return sampled_ids, log_probabilities, lengths

        return sampled_ids


class MLP_RNN_ThreeLSTM(nn.Module):
    def __init__(self, opt):  # (embed_size, hidden_size, vocab_size, num_layers, max_seq_length=50):
        """Set the hyper-parameters and build the layers."""
        super(MLP_RNN_ThreeLSTM, self).__init__()
        self.linear1_1 = nn.Linear(opt.resSize, opt.embed_size)  # image feature is fed into the second LSTM
        self.word_embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        ## LSTM setting
        lstm1_input_size = opt.embed_size + opt.embed_size
        self.lstm1 = nn.LSTM(lstm1_input_size, opt.hidden_size, batch_first=True)
        lstm2_input_size = opt.fc2_size + opt.embed_size
        self.lstm2 = nn.LSTM(lstm2_input_size, opt.hidden_size, batch_first=True)
        lstm3_input_size = opt.attSize + opt.embed_size
        self.lstm3 = nn.LSTM(lstm3_input_size, opt.hidden_size, batch_first=True)

        self.linear2 = nn.Linear(opt.hidden_size*3, opt.vocab_size)
        self.max_seg_length = opt.max_seq_length
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear1_1.weight.data.uniform_(-0.1, 0.1)
        self.linear1_1.bias.data.fill_(0)
        self.linear2.weight.data.uniform_(-0.1, 0.1)
        self.linear2.bias.data.fill_(0)

    def forward(self, img_res, img_embed, img_binary, wordID, lengths):
        # embed word vectors
        embeddings = self.word_embed(wordID)
        # LSTM 1
        img_res = self.relu(self.linear1_1(img_res))
        img_res = img_res.unsqueeze(1)
        img_res = img_res.expand(-1, self.max_seg_length, -1)
        lstm1_input = torch.cat((img_res, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm1_input, lengths, batch_first=True)
        lstm1_hiddens, _ = self.lstm1(packed_embed)
        lstm1_hiddens = self.dropout(lstm1_hiddens[0])

        # LSTM 2
        img_embed = img_embed.unsqueeze(1)
        img_embed = img_embed.expand(-1, self.max_seg_length, -1)
        lstm2_input = torch.cat((img_embed, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm2_input, lengths, batch_first=True)
        lstm2_hiddens, _ = self.lstm2(packed_embed)
        lstm2_hiddens = self.dropout(lstm2_hiddens[0])

        # LSTM 3
        img_binary = img_binary.unsqueeze(1)
        img_binary = img_binary.expand(-1, self.max_seg_length, -1)
        lstm3_input = torch.cat((img_binary, embeddings), 2)
        packed_embed = pack_padded_sequence(lstm3_input, lengths, batch_first=True)
        lstm3_hiddens, _ = self.lstm3(packed_embed)
        lstm3_hiddens = self.dropout(lstm3_hiddens[0])

        lstm_hiddens = torch.cat((lstm1_hiddens, lstm2_hiddens, lstm3_hiddens), 1)
        outputs = self.linear2(lstm_hiddens)

        #unpacked_lstm1_hiddens, _ = pad_packed_sequence(lstm1_hiddens, batch_first=True)
        #unpacked_lstm2_hiddens, _ = pad_packed_sequence(lstm2_hiddens, batch_first=True)
        #unpacked_lstm3_hiddens, _ = pad_packed_sequence(lstm3_hiddens, batch_first=True)

        return outputs

    def generate_sentence(self, img_res, img_embed, img_binary, start_word, end_word, states=(None,None,None), max_sampling_length=30, sample=False):
        sampled_ids = []
        embedded_word = self.word_embed(start_word) # start_word is [1, 1] ?
        img_res = self.relu(self.linear1_1(img_res))
        img_res = img_res.unsqueeze(1)
        img_embed = img_embed.unsqueeze(1)
        img_binary = img_binary.unsqueeze(1)

        embedded_word = embedded_word.expand(img_res.size(0), -1, -1)

        lstm1_states, lstm2_states, lstm3_states = states

        end_word = end_word.squeeze().expand(img_res.size(0))
        reached_end = torch.zeros_like(end_word.data).byte()
        if sample:
            log_probabilities = []
            lengths = torch.zeros_like(reached_end).long()

        i = 0
        while not reached_end.all() and i < max_sampling_length:
            # LSTM 1
            lstm1_input = torch.cat((img_res, embedded_word), 2)  # start with <start_word>
            lstm1_hiddens, lstm1_states = self.lstm1(lstm1_input, lstm1_states)

            # LSTM 2
            lstm2_input = torch.cat((img_embed, embedded_word), 2)
            lstm2_hiddens, lstm2_states = self.lstm2(lstm2_input, lstm2_states)

            # LSTM 2
            lstm3_input = torch.cat((img_binary, embedded_word), 2)
            lstm3_hiddens, lstm3_states = self.lstm3(lstm3_input, lstm3_states)

            lstm_hiddens = torch.cat((lstm1_hiddens.squeeze(1), lstm2_hiddens.squeeze(1), lstm3_hiddens.squeeze(1)), 1)
            outputs = self.linear2(lstm_hiddens)

            if sample:
                predicted, log_p = self.sample(outputs)
                active_batches = (~reached_end)
                log_p *= active_batches.float().to(log_p.device)
                log_probabilities.append(log_p.unsqueeze(1))
                lengths += active_batches.long()
            else:
                predicted = outputs.max(1)[1]
            reached_end = reached_end | predicted.eq(end_word).data
            sampled_ids.append(predicted.unsqueeze(1))
            embedded_word = self.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()
        if sample:
            log_probabilities = torch.cat(log_probabilities, 1).squeeze()
            return sampled_ids, log_probabilities, lengths

        return sampled_ids

