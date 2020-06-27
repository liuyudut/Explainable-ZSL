from __future__ import print_function
import argparse
import os
import random
from sklearn import preprocessing
import pickle
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as func
import math
import numpy as np
import util
import sys
import model_New
import json
from Eval.pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
# CUDA_VISIBLE_DEVICES=0 python clswgan.py --manualSeed 806 --cls_weight 0.1 --syn_num 300 --preprocessing True --val_every 1
# --cuda True --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 97 --ngh 4096 --ndh 4096
# --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 16 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --outname flowers

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='FLO')
parser.add_argument('--dataroot', default='./Datasets', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--train_wordID', default='trainval_wordID')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', default=False)
parser.add_argument('--validation', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netC', default='', help="path to netC (to continue training)")
parser.add_argument('--netR', default='', help="path to netR (to continue training)")
parser.add_argument('--netE', default='', help="path to netE (to continue training)")
parser.add_argument('--netG_name', default='MLP_G')
parser.add_argument('--netD_name', default='MLP_CRITIC')
parser.add_argument('--netC_name', default='MLP_Classify')
parser.add_argument('--netR_name', default='MLP_RNN')
parser.add_argument('--netE_name', default='MLP_E')
parser.add_argument('--save_path', default='./Checkpoint/New_Embed', help='folder to output data and model checkpoints')
parser.add_argument('--save_name', default='CUB', help='folder to output data and model checkpoints')
parser.add_argument('--save_step', type=int, default=40, help='step size for saving trained models')
parser.add_argument('--print_step', type=int, default=1, help='step size for printing log info')
parser.add_argument('--val_step', type=int, default=1, help='step size for validating the model')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed') # CUB:3483 FLO:806 SUN: 4115 AWA1:9182 APY:724
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
## arguments for RNN
parser.add_argument('--vocab_size', type=int, default=1283, help='size of words vocabulary')
parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
parser.add_argument('--lstm_layers', type=int, default=2, help='number of layers in lstm')
parser.add_argument('--max_seq_length', type=int, default=20, help='max_seq_length') ## longer length???
parser.add_argument('--vocab_path', type=str, default='cub_trainval_vocab_thre_4', help='path for saving vocabulary wrapper')
##
parser.add_argument('--neg_num', type=int, default=4, help='size of words vocabulary')
parser.add_argument('--E_sim_weight', type=float, default=1.0, help='weight of the classification loss')
parser.add_argument('--E_rank_weight', type=float, default=1.0, help='weight of the classification loss')
parser.add_argument('--E_cls_weight', type=float, default=1.0, help='weight of the classification loss')
parser.add_argument('--fc1_size', type=int, default=4096, help='the first layer in DEN')
parser.add_argument('--fc2_size', type=int, default=512, help='the second layer in DEN')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.save_path)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# cudnn.benchmark may help increase the running time.
cudnn.benchmark = True

ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data, this is a class instance
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
opt.ntrain_class = data.ntrain_class
# initialize image-text matching network
netE = model_New.MLP_OneWay_Att_Confuse(opt, opt.ntrain_class) ## ! important
if opt.netE != '':
    netE.load_state_dict(torch.load(opt.netE))
netE.apply(model_New.weights_init)
print(netE)

netR = model_New.MLP_RNN_ThreeLSTM(opt)
if opt.netR != '':
    netR.load_state_dict(torch.load(opt.netR))
#netR.apply(model_New.weights_init)
print(netR)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()
lstm_criterion = nn.CrossEntropyLoss()
## loss for image-text matching
sim_criterion = nn.CosineEmbeddingLoss()
reconst_criterion = nn.MSELoss()
binary_criterion = nn.BCEWithLogitsLoss()
##
input_wordID = torch.LongTensor(opt.batch_size, opt.max_seq_length) ## assume max of sentence length is 30
target_wordID = torch.LongTensor(opt.batch_size, opt.max_seq_length) ## assume max of sentence length is 30
input_img_index = torch.LongTensor(opt.batch_size)
input_cap_len = torch.LongTensor(opt.batch_size)
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_att_confuse = torch.FloatTensor(opt.batch_size, opt.nclass_all)
input_att_binary = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)
one = torch.FloatTensor([1])
mone = one * -1
one_batch = torch.ones(opt.batch_size)
# noise_cls = torch.FloatTensor(opt.nclass_all, opt.nz)
# attribute_cls = torch.FloatTensor(opt.nclass_all, opt.attSize)
# attribute_cls = data.attribute
start_word = torch.LongTensor([1])
end_word = torch.LongTensor([2])
start_word = start_word.unsqueeze(0)
end_word = end_word.unsqueeze(0)
#train_img_embed = torch.FloatTensor(data.ntrain, opt.embed_size)
#test_img_embed = torch.FloatTensor(data.ntest_unseen, opt.embed_size)
#test_att_embed = torch.FloatTensor(data.ntest_class, opt.embed_size)

if opt.cuda:
    # netD.cuda()
    # netG.cuda()
    netR.cuda()
    netE.cuda()
    input_wordID = input_wordID.cuda()
    target_wordID = target_wordID.cuda()
    input_img_index = input_img_index.cuda()
    input_cap_len = input_cap_len.cuda()
    input_res = input_res.cuda()
    input_att = input_att.cuda()
    input_att_confuse = input_att_confuse.cuda()
    input_att_binary = input_att_binary.cuda()
    input_label = input_label.cuda()
    one = one.cuda()
    mone = mone.cuda()
    one_batch = one_batch.cuda()
    cls_criterion.cuda()
    lstm_criterion.cuda()
    reconst_criterion.cuda()
    #noise_cls = noise_cls.cuda()
    #attribute_cls = attribute_cls.cuda()
    #data.allclasses = data.allclasses.cuda()
    start_word = start_word.cuda()
    end_word = end_word.cuda()
    ##
    sim_criterion.cuda()
    binary_criterion.cuda()
    data.attribute = data.attribute.cuda()
    data.att_unseen = data.att_unseen.cuda()
    data.att_seen = data.att_seen.cuda()
################

if opt.standardization:
    print('standardization...')
    scaler = preprocessing.StandardScaler()
else:
    scaler = preprocessing.MinMaxScaler()

def matrix_minmax_norm(mat):
    min_v, min_idx = torch.min(mat, 1)
    range_v = torch.max(mat, 1)[0] - min_v
    mat_norm = (torch.transpose(mat, 0, 1) - min_v.repeat(mat.size(1), 1)) / range_v.repeat(mat.size(1), 1)
    return torch.transpose(mat_norm, 0, 1)

def eval_explanation(netR, input_res, image_embed, vocab):
    generated_captions = []
    lstm1_states = (Variable(torch.zeros(1, image_embed.size(0), opt.hidden_size)), Variable(torch.zeros(1, image_embed.size(0), opt.hidden_size)))
    lstm2_states = (Variable(torch.zeros(1, image_embed.size(0), opt.hidden_size)), Variable(torch.zeros(1, image_embed.size(0), opt.hidden_size)))
    if opt.cuda:
        lstm1_states = [s.cuda() for s in lstm1_states] #states = (lstm1_states,lstm2_states),
        lstm2_states = [s.cuda() for s in lstm2_states]

    outputs = netR.generate_sentence(input_res, image_embed, start_word, end_word, states=(lstm1_states,lstm2_states), max_sampling_length=30, sample=False)
    for out_idx in range(len(outputs)):
        sentence = []
        for w in outputs[out_idx]:
            word = vocab[w.data.item()]
            if word != '<end>':
                sentence.append(word)
            else:
                break
        generated_captions.append({"caption": ' '.join(sentence)})

    return generated_captions

def compute_sim_loss(img_embed, text_embed):
    sim_loss = sim_criterion(img_embed, text_embed, one_batch)

    return sim_loss


def compute_ranking_loss(img_embed, text_embed, cls_label):
    alpha1 = 1
    alpha2 = 1
    # hard_ratio = self.hard_ratio
    # rand_ratio = self.rand_radio
    margin = 0.1
    # hard_num = neg_num * hard_ratio
    # rand_num = neg_num * rand_ratio
    num = opt.batch_size
    neg_num = 8 # half of batch size
    img_num = num
    text_num = num
    cnt = num * neg_num
    # calculate distance
    dis_data = torch.matmul(img_embed, torch.transpose(text_embed, 0, 1))
    # smaller is more similar
    dis_data = 1.0 - dis_data  # shape: bs * bs
    # extract distances of paired image-text
    dis_diag = torch.diag(dis_data) # shape: bs
    dis_diag = dis_diag.unsqueeze(1) # shape  bs * 1
    dis_diag_tile = dis_diag.repeat(1, num)
    ## bs * bs distance mat
    i2t_mat = alpha1 * torch.max(torch.zeros_like(dis_data), dis_diag_tile - dis_data + margin)
    t2i_mat = alpha2 * torch.max(torch.zeros_like(dis_data), dis_diag_tile - torch.transpose(dis_data, 0, 1) + margin)
    # find the same paired labels
    rank_loss = torch.zeros([1])
    if opt.cuda:
        rank_loss = rank_loss.cuda()

    label_tile = cls_label.repeat(num, 1) # shape: bs * bs
    # find where are the positive samples
    i2t_pos = torch.where(label_tile == torch.transpose(label_tile, 0, 1), i2t_mat - 100.0, i2t_mat) ## process all samoles in batch size ?? i2t_mat - 100.0 --> 0.0 ??
    t2i_pos = torch.where(label_tile == torch.transpose(label_tile, 0, 1), t2i_mat - 100.0, t2i_mat)
    # find the negative loss
    i2t_values, i2t_index = torch.topk(i2t_pos, neg_num, dim=1, largest=True, sorted=True)
    t2i_values, t2i_index = torch.topk(t2i_pos, neg_num, dim=1, largest=True, sorted=True)
    rank_loss = torch.div(torch.sum(i2t_values) + torch.sum(t2i_values), float(cnt))

    return rank_loss

def compute_diff_loss(img_embed_com_l2, img_embed_spe_l2):
    img_diff = torch.matmul(torch.transpose(img_embed_com_l2, 0, 1), img_embed_spe_l2)
    img_diff_loss = torch.norm(img_diff, p='fro')
    diff_loss = img_diff_loss.mean()

    return diff_loss


def compute_reconst_loss(input_att, text_reconst):
    text_reconst_loss = reconst_criterion(input_att, text_reconst)
    text_reconst_loss = text_reconst_loss.mean()
    reconst_loss = text_reconst_loss

    return reconst_loss

def eval_trainset(train_X, train_label, target_classes, batch_size):
    # extract image embed for unseen classes
    ntest = train_X.size()[0]
    nclass = target_classes.size(0)
    test_img_embed = torch.FloatTensor(ntest, opt.fc2_size)
    test_img_cls = torch.FloatTensor(ntest, opt.ntrain_class)
    #test_text_embed = torch.FloatTensor(nclass, opt.embedSize)
    start = 0
    for ii in range(0, ntest, batch_size):
        end = min(ntest, start + batch_size)
        test_feature = train_X[start:end]
        att_empty_1 = torch.zeros(end - start, opt.attSize)
        att_empty_2 = torch.zeros(end - start, opt.nclass_all)
        if opt.cuda:
            test_feature = test_feature.cuda()
            att_empty_1 = att_empty_1.cuda()
            att_empty_2 = att_empty_2.cuda()
        img_embed_com, _, img_cls, _, _ = netE(test_feature, att_empty_1, att_empty_2)
        img_embed_com_l2 = func.normalize(img_embed_com, p=2, dim=1)
        test_img_embed[start:end, :] = img_embed_com_l2.data.cpu()
        #test_img_cls[start:end, :] = img_cls.data.cpu()
        start = end

    acc_softmax = 0.0

    ## extrat attribute embeddings of seen classes
    res_empty = torch.zeros(nclass, opt.resSize)
    if opt.cuda:
        att_feature = data.att_seen.cuda()
        att_confuse_feature = data.att_confuse_seen.cuda()
        res_empty = res_empty.cuda()

    _, text_embed_com, _, _, _ = netE(res_empty, att_feature, att_confuse_feature)
    text_embed_com_l2 = func.normalize(text_embed_com, p=2, dim=1)
    test_text_embed = text_embed_com_l2.data.cpu()

    # KNN: compute sim matrix
    dis_data = torch.matmul(test_img_embed, torch.transpose(test_text_embed, 0, 1))
    _, predicted_label = torch.max(dis_data, 1)
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    test_label = util.map_label(train_label, target_classes)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx], dtype=torch.float) / torch.sum(idx)

    acc_knn = acc_per_class.mean()

    return acc_softmax, acc_knn


def eval_zsl(test_X, test_label, target_classes, batch_size):
    # extract image embed for unseen classes
    ntest = test_X.size()[0]
    nclass = target_classes.size(0)
    test_img_embed = torch.FloatTensor(ntest, opt.fc2_size)
    #test_text_embed = torch.FloatTensor(nclass, opt.embedSize)
    start = 0
    for ii in range(0, ntest, batch_size):
        end = min(ntest, start + batch_size)
        test_feature = test_X[start:end]
        att_empty_1 = torch.zeros(end - start, opt.attSize)
        att_empty_2 = torch.zeros(end - start, opt.nclass_all)
        if opt.cuda:
            test_feature = test_feature.cuda()
            att_empty_1 = att_empty_1.cuda()
            att_empty_2 = att_empty_2.cuda()
        img_embed_com, _, img_cls, _, _ = netE(test_feature, att_empty_1, att_empty_2)
        img_embed_com_l2 = func.normalize(img_embed_com, p=2, dim=1)
        test_img_embed[start:end, :] = img_embed_com_l2.data.cpu()
        start = end

    ## extrat attribute embeddings of unseen classes
    res_empty = torch.zeros(nclass, opt.resSize)
    if opt.cuda:
        att_feature = data.att_unseen.cuda()
        att_confuse_feature = data.att_confuse_unseen.cuda()
        res_empty = res_empty.cuda()

    _, text_embed_com, _, _, _ = netE(res_empty, att_feature, att_confuse_feature)
    text_embed_com_l2 = func.normalize(text_embed_com, p=2, dim=1)
    test_text_embed = text_embed_com_l2.data.cpu()

    # compute sim matrix
    # KNN: compute sim matrix
    dis_data = torch.matmul(test_img_embed, torch.transpose(test_text_embed, 0, 1))
    _, predicted_label = torch.max(dis_data, 1)

    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    test_label = util.map_label(test_label, target_classes)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx], dtype=torch.float) / torch.sum(idx)

    acc = acc_per_class.mean()

    return acc

def eval_gzsl(test_X, test_label, nclass, target_classes, batch_size):
    # extract image embed for unseen classes
    ntest = test_X.size()[0]
    #nclass = target_classes.size(0)
    test_img_embed = torch.FloatTensor(ntest, opt.fc2_size)
    #test_text_embed = torch.FloatTensor(nclass, opt.embedSize)
    start = 0
    for ii in range(0, ntest, batch_size):
        end = min(ntest, start + batch_size)
        test_feature = test_X[start:end]
        att_empty_1 = torch.zeros(end - start, opt.attSize)
        att_empty_2 = torch.zeros(end - start, opt.nclass_all)
        if opt.cuda:
            test_feature = test_feature.cuda()
            att_empty_1 = att_empty_1.cuda()
            att_empty_2 = att_empty_2.cuda()
        img_embed_com, _, img_cls, _, _ = netE(test_feature, att_empty_1, att_empty_2)
        img_embed_com_l2 = func.normalize(img_embed_com, p=2, dim=1)
        test_img_embed[start:end, :] = img_embed_com_l2.data.cpu()
        start = end

    ## extrat attribute embeddings of unseen classes
    res_empty = torch.zeros(nclass, opt.resSize)
    if opt.cuda:
        att_feature = data.attribute.cuda()
        att_confuse_feature = data.att_confuse.cuda()
        res_empty = res_empty.cuda()

    _, text_embed_com, _, _, _ = netE(res_empty, att_feature, att_confuse_feature)
    text_embed_com_l2 = func.normalize(text_embed_com, p=2, dim=1)
    test_text_embed = text_embed_com_l2.data.cpu()

    # compute sim matrix
    dis_data = torch.matmul(test_img_embed, torch.transpose(test_text_embed, 0, 1))
    _, predicted_label = torch.max(dis_data, 1)

    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        acc_per_class += torch.sum(test_label[idx] == predicted_label[idx], dtype=torch.float) / torch.sum(idx)
    acc_per_class /= target_classes.size(0)

    return acc_per_class


def eval_unseen_classes(netR, data, opt, batch_size=8):

    start_word = torch.LongTensor([1])
    end_word = torch.LongTensor([2])
    start_word = start_word.unsqueeze(0)
    end_word = end_word.unsqueeze(0)
    if opt.cuda:
        start_word = start_word.cuda()
        end_word = end_word.cuda()

    #test_unseen_feature = data.test_unseen_feature
    generated_explain = []
    start = 0
    for i in range(0, data.ntest_unseen, batch_size):
        end = min(data.ntest_unseen, start+ batch_size)
        test_feature = data.test_unseen_feature[start:end]
        att_empty_E1 = torch.zeros(end - start, opt.attSize)
        att_empty_E2 = torch.zeros(end - start, opt.nclass_all)
        lstm1_states = (Variable(torch.zeros(1, end - start, opt.hidden_size)),
                        Variable(torch.zeros(1, end - start, opt.hidden_size)))
        lstm2_states = (Variable(torch.zeros(1, end - start, opt.hidden_size)),
                        Variable(torch.zeros(1, end - start, opt.hidden_size)))
        lstm3_states = (Variable(torch.zeros(1, end - start, opt.hidden_size)),
                        Variable(torch.zeros(1, end - start, opt.hidden_size)))

        start = end
        if opt.cuda:
            test_feature = test_feature.cuda()
            att_empty_E1 = att_empty_E1.cuda()
            att_empty_E2 = att_empty_E2.cuda()
            lstm1_states = [s.cuda() for s in lstm1_states]  # states = (lstm1_states,lstm2_states),
            lstm2_states = [s.cuda() for s in lstm2_states]
            lstm3_states = [s.cuda() for s in lstm3_states]

            # netE1
        with torch.no_grad():
            img_embed, _, _, img_binary, _ = netE(test_feature, att_empty_E1, att_empty_E2)

        # generate words
        img_binary = torch.sigmoid(img_binary)
        outputs = netR.generate_sentence(test_feature, img_embed, img_binary, start_word, end_word, states=(lstm1_states, lstm2_states, lstm3_states),
                               max_sampling_length=20, sample=False)
        for out_idx in range(len(outputs)):
            sentence = []
            for w in outputs[out_idx]:
                word = data.vocab[w.data.item()]
                if word != '<end>':
                    sentence.append(word)
                else:
                    break
            generated_explain.append({"annotations": ' '.join(sentence)})

    return generated_explain


def eval_seen_classes(netR, data, opt, batch_size=8):

    start_word = torch.LongTensor([1])
    end_word = torch.LongTensor([2])
    start_word = start_word.unsqueeze(0)
    end_word = end_word.unsqueeze(0)
    if opt.cuda:
        start_word = start_word.cuda()
        end_word = end_word.cuda()

    generated_explain = []
    start = 0
    for i in range(0, data.ntest_seen, batch_size):
        end = min(data.ntest_seen, start+ batch_size)
        test_feature = data.test_seen_feature[start:end]
        att_empty_E1 = torch.zeros(end - start, opt.attSize)
        att_empty_E2 = torch.zeros(end - start, opt.nclass_all)
        lstm1_states = (Variable(torch.zeros(1, end - start, opt.hidden_size)),
                        Variable(torch.zeros(1, end - start, opt.hidden_size)))
        lstm2_states = (Variable(torch.zeros(1, end - start, opt.hidden_size)),
                        Variable(torch.zeros(1, end - start, opt.hidden_size)))
        lstm3_states = (Variable(torch.zeros(1, end - start, opt.hidden_size)),
                        Variable(torch.zeros(1, end - start, opt.hidden_size)))

        start = end
        if opt.cuda:
            test_feature = test_feature.cuda()
            att_empty_E1 = att_empty_E1.cuda()
            att_empty_E2 = att_empty_E2.cuda()
            lstm1_states = [s.cuda() for s in lstm1_states]  # states = (lstm1_states,lstm2_states),
            lstm2_states = [s.cuda() for s in lstm2_states]
            lstm3_states = [s.cuda() for s in lstm3_states]

            # netE1
        with torch.no_grad():
            img_embed, _, _, img_binary, _ = netE(test_feature, att_empty_E1, att_empty_E2)

        # generate words
        img_binary = torch.sigmoid(img_binary)
        outputs = netR.generate_sentence(test_feature, img_embed, img_binary, start_word, end_word, states=(lstm1_states, lstm2_states, lstm3_states),
                               max_sampling_length=20, sample=False)
        for out_idx in range(len(outputs)):
            sentence = []
            for w in outputs[out_idx]:
                word = data.vocab[w.data.item()]
                if word != '<end>':
                    sentence.append(word)
                else:
                    break
            generated_explain.append({"annotations": ' '.join(sentence)})

    return generated_explain


def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

def save_json(json_dict, save_name):
  with open(save_name, 'w') as outfile:
    json.dump(json_dict, outfile)
  print("Wrote json file to %s" % save_name)


###############################################################################################
print("Number of overlapping classes between trainval and test:",len(set(data.seenclasses).intersection(set(data.unseenclasses))))
print("Starting to train loop...")
vis = util.Visualizer(env='Explain')
# setup optimizer
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# ## RNN updates Generator as well
# optimizerR = optim.Adam(list(netG.parameters()) + list(netR.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
netE.load_state_dict(torch.load(os.path.join(opt.save_path, 'CUB_OneWay_Embed_New:netE_Epoch-199.ckpt')))
optimizerE = optim.Adam(netE.parameters(), lr=0.000005, betas=(opt.beta1, 0.999)) # smaller LR, more epochs lr=0.000005
#optimizerR = optim.Adam(list(netE.parameters()) + list(netR.parameters()), lr=0.000005, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(list(netE.parameters()) + list(netR.parameters()), lr=0.000005, betas=(opt.beta1, 0.999))
##
best_acc_zsl = 0
best_acc_unseen_gzsl = 0
best_acc_seen_gzsl = 0
best_H_gzsl = 0
# # freeze the classifier during the optimization
# for p in pretrain_cls.model.parameters():  # set requires_grad to False
#     p.requires_grad = False
#total_feature = torch.cat((data.train_feature, syn_feature), 0)
#total_label = torch.cat((data.train_label, syn_label), 0)
ntotal = data.ntrain_caption
opt.n_batch = ntotal // opt.batch_size
#total_feature = data.train_feature
#total_label = data.train_label
for epoch in range(opt.nepoch):
    FP = 0  ## what is this?
    mean_sim_loss = 0.0
    mean_cls_loss = 0.0
    mean_rank_loss = 0.0
    mean_binary_loss = 0.0
    mean_reconst_loss = 0.0
    mean_R_loss = 0.0
    mean_C_loss = 0.0
    ## random shuffle
    idx = torch.randperm(ntotal)
    data.train_input_wordID = data.train_input_wordID[idx]
    data.train_target_wordID = data.train_target_wordID[idx]
    data.train_img_index = data.train_img_index[idx]
    data.train_cap_len = data.train_cap_len[idx]
    #data.attribute = data.attribute[idx]
    ##
    for i in range(opt.n_batch):
        #print(i)
        start_i = i * opt.batch_size
        end_i = start_i + opt.batch_size
        # batch data in cpu
        batch_input_wordID = data.train_input_wordID[start_i:end_i]
        batch_target_wordID = data.train_target_wordID[start_i:end_i]
        batch_img_index = data.train_img_index[start_i:end_i]
        batch_cap_len = data.train_cap_len[start_i:end_i]
        # sort mini-batch based on the caption length
        sorted_seq_lengths, indices = torch.sort(batch_cap_len, descending=True)
        batch_input_wordID = batch_input_wordID[indices]
        batch_target_wordID = batch_target_wordID[indices]
        batch_img_index = batch_img_index[indices]
        batch_cap_len = batch_cap_len[indices]
        ## get one batch
        batch_feature = data.train_feature[batch_img_index]
        batch_label = data.train_label[batch_img_index]
        batch_att = data.attribute[batch_label]
        batch_att_binary = data.att_binary[batch_label]
        batch_att_confuse = data.att_confuse[batch_label]
        # copy data to gpu
        input_wordID.copy_(batch_input_wordID)
        target_wordID.copy_(batch_target_wordID)
        input_img_index.copy_(batch_img_index)
        input_cap_len.copy_(batch_cap_len)
        input_res.copy_(batch_feature)
        input_att.copy_(batch_att)
        input_att_confuse.copy_(batch_att_confuse)
        input_label.copy_(util.map_label(batch_label, data.seenclasses))
        input_att_binary.copy_(batch_att_binary)

        # Train Embedding network
        img_embed, text_embed, img_cls, img_binary, img_reconst = netE(input_res, input_att, input_att_confuse)
        img_embed_l2 = func.normalize(img_embed, p=2, dim=1)
        text_embed_l2 = func.normalize(text_embed, p=2, dim=1)

        E_sim_cost = compute_sim_loss(img_embed_l2, text_embed_l2)
        E_cls_cost = cls_criterion(img_cls, input_label)
        E_binary_cost = binary_criterion(img_binary, input_att_binary)
        E_reconst_cost = compute_reconst_loss(img_reconst, input_res)
        #E_diff_cost = compute_diff_loss(img_diff_l2, img_cls_l2)
        E_cost = 1.0 * E_cls_cost + 1.0 * E_sim_cost + 1.0 * E_binary_cost + 1.0 * E_reconst_cost
        ##
        optimizerE.zero_grad()
        E_cost.backward(retain_graph=True)
        optimizerE.step()
        mean_sim_loss = E_sim_cost.item()
        mean_cls_loss = E_cls_cost.item()
        mean_binary_loss = E_binary_cost.item()
        mean_reconst_loss = E_reconst_cost.item()

        ##################################
        # (2) Train Explanation network
        ##################################
        img_binary = torch.sigmoid(img_binary)
        lstm_outputs = netR(input_res, img_embed, img_binary, input_wordID, input_cap_len)
        lstm_targets = pack_padded_sequence(target_wordID, input_cap_len, batch_first=True)[0]
        # LSTM loss
        lstm_cost = lstm_criterion(lstm_outputs, lstm_targets)
        R_cost = lstm_cost
        optimizerR.zero_grad()
        R_cost.backward()
        optimizerR.step()
        mean_R_loss = lstm_cost.item()
        # evaluate mode
        # netE.eval()
    # Generalized zero-shot learning
    print('[%d/%d] R_loss: %.4f' % (epoch, opt.nepoch, mean_R_loss))
    # Generate sentence
    # generated_exp = eval_explanation(netR, input_res, data.vocab) ## why it is an error??
    # print(generated_exp[0]['caption'])
    if (epoch + 1) % 10 == 0:
        # set evaluation
        netE.eval()
        netR.eval()

        # evaluate netE
        train_acc_softmax, train_acc_knn = eval_trainset(data.train_feature, data.train_label, data.seenclasses, 100)
        test_acc_seen = eval_gzsl(data.test_seen_feature, data.test_seen_label, data.allclasses.size(0), data.seenclasses, 100)
        test_acc_unseen = eval_gzsl(data.test_unseen_feature, data.test_unseen_label, data.allclasses.size(0), data.unseenclasses, 100)

        test_acc_knn = eval_zsl(data.test_unseen_feature, data.test_unseen_label, data.unseenclasses, 100)

        if (test_acc_seen + test_acc_unseen) > 0:
            H = 2 * test_acc_seen * test_acc_unseen / (test_acc_seen + test_acc_unseen)
        #print('%.4f %.4f %.4f' % (test_acc_unseen, test_acc_seen, H)) , 'E_reconst_cost': mean_reconst_loss
        if H > best_H_gzsl:
            best_H_gzsl = H
            best_acc_unseen_gzsl = test_acc_unseen
            best_acc_seen_gzsl = test_acc_seen

        if test_acc_knn > best_acc_zsl:
            best_acc_zsl = test_acc_knn

        print(
            '[%d/%d] mean_sim_loss: %0.4f mean_cls_loss: %.4f mean_binary_loss: %0.4f mean_reconst_loss: %0.4f' % (
            epoch, opt.nepoch, mean_sim_loss, mean_cls_loss, mean_binary_loss, mean_reconst_loss))
        #
        print('[%d/%d] Train_Softmax: %0.4f Train_KNN: %.4f Test_unseen_ZSL: %0.4f Test_seen_GZSL: %0.4f Test_unseen_GZSL: %0.4f H: %.4f' % (epoch, opt.nepoch, train_acc_softmax, train_acc_knn, test_acc_knn, test_acc_seen, test_acc_unseen, H))
        #
        # evaluate netR for test unseen classes
        print("generate the explanation of test unseen classes")
        generated_explain = eval_unseen_classes(netR, data, opt, batch_size=64)
        # save generate results and ground-truth results
        json_results = []
        test_unseen_images = read_json('./Datasets/CUB/descriptions_bird.test_unseen_images.json')
        for i in range(data.ntest_unseen):
            img_dict = {'image_id': test_unseen_images['images'][i]['id'],
                        'caption': generated_explain[i]['annotations']}
            json_results.append(img_dict)
        ## generated annotations
        json_path = './Results/Explain_captions/CUB_OneWay_fixed_only_image_embed_test_unseen.json'
        save_json(json_results, json_path)
        ## ground truth annotations
        coco = COCO("./Datasets/CUB/descriptions_bird.test_unseen_images.json")
        cocoRes = coco.loadRes(json_path)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.evaluate()
        print(cocoEval.eval.items())

        # reset training mode
        netE.train()
        netR.train()


        torch.save(netE.state_dict(), os.path.join(opt.save_path, opt.save_name + '_OneWay_Embed_Explain_Joint_Good:netE_Epoch-{}.ckpt'.format(epoch)))
        torch.save(netR.state_dict(),os.path.join(opt.save_path, opt.save_name + '_OneWay_Embed_Explain_Joint_Good:netR_Epoch-{}.ckpt'.format(epoch)))

print('GZSL best unseen=%.4f, seen=%.4f, h=%.4f' % (best_acc_unseen_gzsl, best_acc_seen_gzsl, best_H_gzsl))
print('ZSL best unseen=%.4f' % (best_acc_zsl))

