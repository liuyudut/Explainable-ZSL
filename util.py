import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import torch.nn as nn
import json
import pickle
from datetime import datetime
import visdom

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

def save_json(json_dict, save_name):
  with open(save_name, 'w') as outfile:
    json.dump(json_dict, outfile)
  print("Wrote json file to %s" % save_name)

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class Visualizations(object):
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.acc_win = None

    def plot_acc(self, acc, step, title='Accuracy', xlabel='epochs', ylabel='Top-1 accuracy'):
        self.acc_win = self.vis.line(
            [acc],
            [step],
            win=self.acc_win,
            update='append' if self.acc_win else None,
            opts=dict(
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
            )
        )

class Visualizer(object):
    def __init__(self, env='default'):
        self.vis = visdom.Visdom(env=env)
        self.acc_index = {}
        self.loss_index = {}

    def plot_acc(self, d, title, xlabel, ylabel):
        '''
        self.plot('loss',1.00)
        '''
        name=list(d.keys())
        name_total=" ".join(name)
        x = self.acc_index.get(name_total, 0)
        val=list(d.values())
        if len(val)==1:
            y=np.array(val)
        else:
            y=np.array(val).reshape(-1,len(val))
        #print(x)
        self.vis.line(Y=y,X=np.ones(y.shape)*x,
                    win=str(name_total),#unicode
                    opts=dict(legend=name, xlabel=xlabel, ylabel=ylabel,title=title),
                    update=None if x == 0 else 'append'
                    )
        self.acc_index[name_total] = x + 1

    def plot_loss(self, d, title, xlabel, ylabel):
        '''
        self.plot('loss',1.00)
        '''
        name=list(d.keys())
        name_total=" ".join(name)
        x = self.loss_index.get(name_total, 0)
        val=list(d.values())
        if len(val)==1:
            y=np.array(val)
        else:
            y=np.array(val).reshape(-1,len(val))
        #print(x)
        self.vis.line(Y=y,X=np.ones(y.shape)*x,
                    win=str(name_total),#unicode
                    opts=dict(legend=name, xlabel=xlabel, ylabel=ylabel,title=title),
                    update=None if x == 0 else 'append'
                    )
        self.loss_index[name_total] = x + 1

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_captions(opt)
                self.read_matdataset(opt)
                self.read_vocab(opt)

        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_vocab(self, opt):
        #self.vocab = read_json(opt.dataroot + "/" + opt.dataset + "/" + opt.vocab_path + ".json")
        self.vocab = read_json(opt.dataroot + "/CUB/" + opt.vocab_path + ".json")

    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()] 
        trainval_loc = fid['trainval_loc'][()] 
        train_loc = fid['train_loc'][()] 
        val_unseen_loc = fid['val_unseen_loc'][()] 
        test_seen_loc = fid['test_seen_loc'][()] 
        test_unseen_loc = fid['test_unseen_loc'][()] 
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc] 
            self.train_label = label[trainval_loc] 
            self.test_unseen_feature = feature[test_unseen_loc] 
            self.test_unseen_label = label[test_unseen_loc] 
            self.test_seen_feature = feature[test_seen_loc] 
            self.test_seen_label = label[test_seen_loc] 
        else:
            self.train_feature = feature[train_loc] 
            self.train_label = label[train_loc] 
            self.test_unseen_feature = feature[val_unseen_loc] 
            self.test_unseen_label = label[val_unseen_loc] 

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()


        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long() 
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long() 
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long() 
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_captions(self, opt):
        #captions = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.train_wordID + ".mat")
        captions = sio.loadmat(opt.dataroot + "/CUB/" + opt.train_wordID + ".mat")
        train_input_wordID = captions['input_wordID']
        train_target_wordID = captions['target_wordID']
        ## set max length = 20
        train_input_wordID = train_input_wordID[:,:20]
        train_target_wordID = train_target_wordID[:,:20]

        train_img_index = (captions['img_index'].T).squeeze()
        train_cap_len = (captions['caption_length'].T).squeeze()
        ## set max length = 20
        train_cap_len = np.minimum(train_cap_len, 20)

        self.train_input_wordID = torch.from_numpy(train_input_wordID).long()
        self.train_target_wordID = torch.from_numpy(train_target_wordID).long()

        self.train_img_index = torch.from_numpy(train_img_index).long()
        self.train_cap_len = torch.from_numpy(train_cap_len).long()
        self.train_cap_len = self.train_cap_len - 1 # exclude <start> in the target sentence or <end> in the input sentence !!
        self.ntrain_caption = self.train_input_wordID.size()[0]

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + "_ours.mat")
        feature = matcontent['features'].T ## matrix transpose
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        # data splits are included in mat file.
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        # numpy to tensor
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        self.att_confuse = torch.matmul(self.attribute, torch.transpose(self.attribute, 0, 1))
        #self.att_confuse = (self.att_confuse + 1
        # read att_agg vectors
        #matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_agg.mat")
        #self.att_agg = torch.from_numpy(matcontent['att_agg'].T).float()
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_binary_top10.mat")
        self.att_binary = torch.from_numpy(matcontent['att_binary'].T).float()
        ## load binary_frequent for all the images
        #matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + "binary_subset.mat")
        #self.binary_frequent = torch.from_numpy(matcontent['binary_subset'].T).float() ## matrix transpose
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                ## only preprocess visual features
                _train_feature = scaler.fit_transform(feature[trainval_loc]) ## why it is fit_transform rather than transform?------------------question
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                # ## L2 Norm
                # _train_feature = preprocessing.normalize(feature[trainval_loc], norm='l2')
                # _test_seen_feature = preprocessing.normalize(feature[test_seen_loc], norm='l2')
                # _test_unseen_feature = preprocessing.normalize(feature[test_unseen_loc], norm='l2')
                self.train_feature = torch.from_numpy(_train_feature).float()
                #mx = self.train_feature.max()
                #self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                #self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                #self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
                # binary_frequent
                # self.train_binary_frequent = torch.from_numpy(binary_frequent[trainval_loc]).float()
                # self.test_unseen_binary_frequent = torch.from_numpy(binary_frequent[test_unseen_loc]).float()
                # self.test_seen_binary_frequent = torch.from_numpy(binary_frequent[test_seen_loc]).float()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy())) ## .long()??
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.att_unseen = self.attribute[np.unique(self.test_unseen_label.numpy()),:]
        self.att_seen = self.attribute[np.unique(self.train_label.numpy()), :]
        #self.att_agg_seen = self.att_agg[np.unique(self.train_label.numpy()), :]
        #self.att_agg_unseen = self.att_agg[np.unique(self.test_unseen_label.numpy()), :]
        self.att_confuse_seen = self.att_confuse[np.unique(self.train_label.numpy()), :]
        self.att_confuse_unseen = self.att_confuse[np.unique(self.test_unseen_label.numpy()), :]
        self.ntrain = self.train_feature.size()[0]  ## careful...
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        if not opt.validation:
            self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.from_numpy(np.unique(label))
        # map {20,101} -> {0,81}
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]] 
    
    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_New(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    # select batch samples by randomly drawing batch_size classes    
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]] 
        return batch_feature, batch_label, batch_att

class CompactBilinearPooling(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, sum_pool=True):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: torch.sparse.FloatTensor(torch.stack([torch.arange(input_dim, out = torch.LongTensor()), rand_h.long()]), rand_s.float(), [input_dim, output_dim]).to_dense()
        self.sketch_matrix1 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim1,)), 2 * torch.randint(2, size = (input_dim1,)) - 1, input_dim1, output_dim))
        self.sketch_matrix2 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim2,)), 2 * torch.randint(2, size = (input_dim2,)) - 1, input_dim2, output_dim))

    def forward(self, x1, x2):
        fft1 = torch.rfft(x1.permute(0, 2, 3, 1).matmul(self.sketch_matrix1), 1)
        fft2 = torch.rfft(x2.permute(0, 2, 3, 1).matmul(self.sketch_matrix2), 1)
        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
        cbp = torch.irfft(fft_product, 1, signal_sizes = (self.output_dim,)) * self.output_dim
        return cbp.sum(dim = 1).sum(dim = 1) if self.sum_pool else cbp.permute(0, 3, 1, 2)