from plot import plot_acc, plot_gan_losses, plot_confusion_matrix
from arguments import parse_args
import random
import torch
import torch.backends.cudnn as cudnn
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from dataset import FeaturesCls, FeaturesGAN
from train_cls import TrainCls
from train_gan import TrainGAN
from generate import load_unseen_att, load_all_att, load_unseen_att_with_bg
from mmdetection.splits import get_unseen_class_labels
from clustercontrast.models.cm import ClusterMemory
import model
opt = parse_args()
import collections
import torch.nn.functional as F


try:
    os.makedirs(opt.outname)
except OSError:
    pass

# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)

for arg in vars(opt): print(f"######################  {arg}: {getattr(opt, arg)}")


print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
#
torch.manual_seed(opt.manualSeed)
#
if opt.cuda:
     torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

unseen_attributes, unseen_att_labels = load_unseen_att(opt)
attributes, _ = load_all_att(opt)
# init classifier
trainCls = TrainCls(opt)

print('initializing GAN Trainer')


start_epoch = 0

seenDataset = FeaturesGAN(opt)

####
##todo memory bank
# def generate_syn_initial_feature(labels, attribute, num=100, no_grad=True):
#     """
#     generates features
#     inputs:
#         labels: features labels to generate nx1 n is number of objects
#         attributes: attributes of objects to generate (nxd) d is attribute dimensions
#         num: number of features to generate for each object
#     returns:
#         1) synthesised features
#         2) labels of synthesised  features
#     """
#
#     nclass = labels.shape[0]
#     syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
#     syn_label = torch.LongTensor(nclass * num)
#
#     syn_att = torch.FloatTensor(num, opt.attSize)
#     syn_noise = torch.FloatTensor(num, opt.nz)
#
#     with torch.no_grad():
#         netG = model.MLP_G(opt)
#         checkpoint = torch.load(opt.pretrain_net_G)
#         netG.load_state_dict(checkpoint['state_dict'])
#         print(f"loaded weights from best model")
#
#         for i in range(nclass):
#             label = labels[i]
#             iclass_att = attribute[i]
#             syn_att.copy_(iclass_att.repeat(num, 1))
#             syn_noise.normal_(0, 1)
#             output = netG(syn_noise, syn_att)
#
#             syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
#             syn_label.narrow(0, i * num, num).fill_(label)
#
#     return syn_feature, syn_label

@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers

#### 21 classes
seen_features = torch.from_numpy(seenDataset.all_features)
seen_lables = seenDataset.all_labels


# unseen_features, unseen_lables = generate_syn_initial_feature(unseen_att_labels, unseen_attributes,
#                                                                    num=opt.syn_num, no_grad=True)
unseen_features = np.load(
    "/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/model/features/memory_bank/voc/GPU4_zero_shot_detection2_VOC_2022_08_15_66.2/unseen_features_4000/test_feats.npy")
unseen_labels = np.load(
    "/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/model/features/memory_bank/voc/GPU4_zero_shot_detection2_VOC_2022_08_15_66.2/unseen_features_4000/test_labels.npy")

init_features = torch.cat((seen_features, torch.from_numpy(unseen_features)), 0)
init_labels = np.concatenate((seen_lables, torch.from_numpy(unseen_labels)))



num_features = init_features.shape[1]
num_cluster = len(np.unique(init_labels))

cluster_features = generate_cluster_features(init_labels, init_features)
##
#np.save('/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/model/features/memory_bank/voc/GPU4_zero_shot_detection2_VOC_2022_08_15_66.2/unseen_features_4000_memory_bank_initial_feats/memory_bank_feats.npy', cluster_features)
##

del seen_features, unseen_features, init_features

# create memory
memory = ClusterMemory(num_features, num_cluster, temp=opt.cm_temp,
                       momentum=opt.cm_momentum, use_hard=opt.cm_use_hard).cuda()
memory.features = F.normalize(cluster_features, dim=1).cuda()
####
unseen_att_with_bg, unseen_att_labels_with_bg = load_unseen_att_with_bg(opt)


trainFGGAN = TrainGAN(opt, attributes, unseen_attributes, unseen_att_labels, memory, unseen_att_with_bg, unseen_att_labels_with_bg, seen_feats_mean=seenDataset.features_mean, gen_type='FG')

if opt.netD and opt.netG:
    start_epoch = trainFGGAN.load_checkpoint()
    
for epoch in range(start_epoch, opt.nepoch):
    features, labels = seenDataset.epochData(include_bg=False)
    # features, labels = seenDataset.epochData(include_bg=True)
    # train GAN
    trainFGGAN(epoch, features, labels)
    # synthesize features
    syn_feature, syn_label = trainFGGAN.generate_syn_feature(unseen_att_labels, unseen_attributes, num=opt.syn_num)
    num_of_bg = opt.syn_num*2

    real_feature_bg, real_label_bg = seenDataset.getBGfeats(num_of_bg)

    # concatenate synthesized + real bg features
    syn_feature = np.concatenate((syn_feature.data.numpy(), real_feature_bg))
    syn_label = np.concatenate((syn_label.data.numpy(), real_label_bg))

    
    trainCls(syn_feature, syn_label, gan_epoch=epoch)

    # -----------------------------------------------------------------------------------------------------------------------
    # plots
    classes = np.concatenate((['background'], get_unseen_class_labels(opt.dataset, split=opt.classes_split)))
    plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Train.npy'), classes, classes, opt, dataset='Train', prefix=opt.class_embedding.split('/')[-1])
    plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Test.npy'), classes, classes, opt, dataset='Test', prefix=opt.class_embedding.split('/')[-1])
    plot_acc(np.vstack(trainCls.val_accuracies), opt, prefix=opt.class_embedding.split('/')[-1])

    # save models
    if trainCls.isBestIter == True:
        trainFGGAN.save_checkpoint(state='best')

    trainFGGAN.save_checkpoint(state='latest')
