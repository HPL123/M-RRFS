from generate import load_unseen_att, load_all_att, load_seen_att
from train_gan import TrainGAN
import torch
from torch.autograd import Variable
from dataset import FeaturesCls, FeaturesGAN
import model
import argparse
import numpy as np

# ##vis unseen:
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', default='voc', help='coco, voc')
#     parser.add_argument('--dataroot', default='../data', help='path to dataset')
#     parser.add_argument('--class_embedding', default='VOC/fasttext_synonym.npy')
#     parser.add_argument('--syn_num', type=int, default=500, help='number features to generate per class')
#
#     parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
#     parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
#     parser.add_argument('--resSize', type=int, default=1024, help='size of visual features')
#     parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
#     parser.add_argument('--nz', type=int, default=300, help='size of the latent z vector')
#     parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
#     parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
#
#     parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
#     parser.add_argument('--netG',
#                         default='/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/code/code_suz_contra/pascal/zero_shot_detection1/checkpoints/VOC_11_6_1/gen_best.pth',
#                         help="path to netG (to continue training)")
#     # parser.add_argument('--netG',
#     #                     default='/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/code/code_suz_contra/pascal/zero_shot_detection2/checkpoints/VOC_11_12_2/gen_best.pth',
#     #                     help="path to netG (to continue training)")
#     parser.add_argument('--classes_split', default='16_14')
#     parser.add_argument('--nclass_all', type=int, default=21, help='number of all classes')
#     parser.add_argument('--save_dir', default='/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/results/visulization/features_vis/voc/features/', help='the dir to save feats and labels')
#     parser.add_argument('--data_split', default='test', help='the dataset train, val, test to load from cfg file')
#
#     args = parser.parse_args()
#     return args
# args = parse_args()
#
#
#
# def generate_syn_feature(labels, attribute, num=100, no_grad=True):
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
#     syn_feature = torch.FloatTensor(nclass * num, args.resSize)
#     syn_label = torch.LongTensor(nclass * num)
#
#     syn_att = torch.FloatTensor(num, args.attSize)
#     syn_noise = torch.FloatTensor(num, args.nz)
#
#     netG = model.MLP_G(args)
#     checkpoint = torch.load(args.netG)
#     netG.load_state_dict(checkpoint['state_dict'])
#     print(f"loaded weights from best model")
#
#     with torch.no_grad():
#         for i in range(nclass):
#             label = labels[i]
#             iclass_att = attribute[i]
#             syn_att.copy_(iclass_att.repeat(num, 1))
#             syn_noise.normal_(0, 1)
#             # output = netG(Variable(syn_noise), Variable(syn_att))
#             output = netG(syn_noise, syn_att)
#
#             syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
#             syn_label.narrow(0, i * num, num).fill_(label)
#
#     return syn_feature, syn_label
#
# unseen_attributes, unseen_att_labels = load_unseen_att(args)
# features, labels = generate_syn_feature(unseen_att_labels, unseen_attributes, args.syn_num)
#
# np.save(f'{args.save_dir}/{args.data_split}_feats.npy', features)
# np.save(f'{args.save_dir}/{args.data_split}_labels.npy', labels)


##vis unseen:
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='voc', help='coco, voc')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--class_embedding', default='VOC/fasttext_synonym.npy')
    parser.add_argument('--syn_num', type=int, default=500, help='number features to generate per class')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--resSize', type=int, default=1024, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=300, help='size of the latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')

    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    # parser.add_argument('--netG',
    #                     default='/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/code/code_suz_contra/pascal/zero_shot_detection1/checkpoints/VOC_11_6_1/gen_best.pth',
    #                     help="path to netG (to continue training)")
    parser.add_argument('--netG',
                        default='/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/code/code_suz_contra/pascal/zero_shot_detection2/checkpoints/VOC_11_12_2/gen_best.pth',
                        help="path to netG (to continue training)")
    parser.add_argument('--classes_split', default='16_14')
    parser.add_argument('--nclass_all', type=int, default=21, help='number of all classes')
    parser.add_argument('--save_dir', default='/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/results/visulization/features_vis/voc_baseline_in_paper/features/', help='the dir to save feats and labels')
    parser.add_argument('--data_split', default='test', help='the dataset train, val, test to load from cfg file')

    args = parser.parse_args()
    return args
args = parse_args()



def generate_syn_feature(labels, attribute, num=100, no_grad=True):
    """
    generates features
    inputs:
        labels: features labels to generate nx1 n is number of objects
        attributes: attributes of objects to generate (nxd) d is attribute dimensions
        num: number of features to generate for each object
    returns:
        1) synthesised features
        2) labels of synthesised  features
    """

    nclass = labels.shape[0]
    syn_feature = torch.FloatTensor(nclass * num, args.resSize)
    syn_label = torch.LongTensor(nclass * num)

    syn_att = torch.FloatTensor(num, args.attSize)
    syn_noise = torch.FloatTensor(num, args.nz)

    netG = model.MLP_G(args)
    checkpoint = torch.load(args.netG)
    netG.load_state_dict(checkpoint['state_dict'])
    print(f"loaded weights from best model")

    with torch.no_grad():
        for i in range(nclass):
            label = labels[i]
            iclass_att = attribute[i]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            # output = netG(Variable(syn_noise), Variable(syn_att))
            output = netG(syn_noise, syn_att)

            syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i * num, num).fill_(label)

    return syn_feature, syn_label

unseen_attributes, unseen_att_labels = load_unseen_att(args)
features, labels = generate_syn_feature(unseen_att_labels, unseen_attributes, args.syn_num)

np.save(f'{args.save_dir}/{args.data_split}_feats.npy', features)
np.save(f'{args.save_dir}/{args.data_split}_labels.npy', labels)


#vis unseen baseline:
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', default='voc', help='coco, voc')
#     parser.add_argument('--dataroot', default='../data', help='path to dataset')
#     parser.add_argument('--class_embedding', default='VOC/fasttext_synonym.npy')
#     parser.add_argument('--syn_num', type=int, default=400, help='number features to generate per class')
#
#     parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
#     parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
#     parser.add_argument('--resSize', type=int, default=1024, help='size of visual features')
#     parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
#     parser.add_argument('--nz', type=int, default=300, help='size of the latent z vector')
#     parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
#     parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
#
#     parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
#     parser.add_argument('--netG',
#                         default='/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/code/code_suz_contra/pascal/zero_shot_detection/checkpoints/VOC_11_13_3/gen_best.pth',
#                         help="path to netG (to continue training)")
#     parser.add_argument('--classes_split', default='16_14')
#     parser.add_argument('--nclass_all', type=int, default=21, help='number of all classes')
#     parser.add_argument('--save_dir', default='/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/results/visulization/features_vis/voc_baseline/features/', help='the dir to save feats and labels')
#     parser.add_argument('--data_split', default='test', help='the dataset train, val, test to load from cfg file')
#
#     args = parser.parse_args()
#     return args
# args = parse_args()
#
#
#
# def generate_syn_feature(labels, attribute, num=100, no_grad=True):
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
#     syn_feature = torch.FloatTensor(nclass * num, args.resSize)
#     syn_label = torch.LongTensor(nclass * num)
#
#     syn_att = torch.FloatTensor(num, args.attSize)
#     syn_noise = torch.FloatTensor(num, args.nz)
#
#     netG = model.MLP_G(args)
#     checkpoint = torch.load(args.netG)
#     netG.load_state_dict(checkpoint['state_dict'])
#     print(f"loaded weights from best model")
#
#     with torch.no_grad():
#         for i in range(nclass):
#             label = labels[i]
#             iclass_att = attribute[i]
#             syn_att.copy_(iclass_att.repeat(num, 1))
#             syn_noise.normal_(0, 1)
#             # output = netG(Variable(syn_noise), Variable(syn_att))
#             output = netG(syn_noise, syn_att)
#
#             syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
#             syn_label.narrow(0, i * num, num).fill_(label)
#
#     return syn_feature, syn_label
#
# unseen_attributes, unseen_att_labels = load_unseen_att(args)
# features, labels = generate_syn_feature(unseen_att_labels, unseen_attributes, args.syn_num)
#
# np.save(f'{args.save_dir}/{args.data_split}_feats.npy', features)
# np.save(f'{args.save_dir}/{args.data_split}_labels.npy', labels)


##vis seen
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', default='voc', help='coco, voc')
#     parser.add_argument('--dataroot', default='../data', help='path to dataset')
#     parser.add_argument('--class_embedding', default='VOC/fasttext_synonym.npy')
#     parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
#
#     parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
#     parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
#     parser.add_argument('--resSize', type=int, default=1024, help='size of visual features')
#     parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
#     parser.add_argument('--nz', type=int, default=300, help='size of the latent z vector')
#     parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
#     parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
#
#     parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
#     parser.add_argument('--netG',
#                         default='/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/code/code_suz_contra/pascal/zero_shot_detection1/checkpoints/VOC_11_6_1/gen_best.pth',
#                         help="path to netG (to continue training)")
#     parser.add_argument('--classes_split', default='16_14')
#     parser.add_argument('--nclass_all', type=int, default=21, help='number of all classes')
#     parser.add_argument('--save_dir', default='/disk4/hpl/zreo_shot_object_detection/SUZ_zero_shot/results/visulization/features_vis/voc/features/', help='the dir to save feats and labels')
#     parser.add_argument('--data_split', default='train', help='the dataset train, val, test to load from cfg file')
#
#     args = parser.parse_args()
#     return args
# args = parse_args()
#
#
#
# def generate_syn_feature(labels, attribute, num=100, no_grad=True):
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
#     syn_feature = torch.FloatTensor(nclass * num, args.resSize)
#     syn_label = torch.LongTensor(nclass * num)
#
#     syn_att = torch.FloatTensor(num, args.attSize)
#     syn_noise = torch.FloatTensor(num, args.nz)
#
#     netG = model.MLP_G(args)
#     checkpoint = torch.load(args.netG)
#     netG.load_state_dict(checkpoint['state_dict'])
#     print(f"loaded weights from best model")
#
#     with torch.no_grad():
#         for i in range(nclass):
#             label = labels[i]
#             iclass_att = attribute[i]
#             syn_att.copy_(iclass_att.repeat(num, 1))
#             syn_noise.normal_(0, 1)
#             # output = netG(Variable(syn_noise), Variable(syn_att))
#             output = netG(syn_noise, syn_att)
#
#             syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
#             syn_label.narrow(0, i * num, num).fill_(label)
#
#     return syn_feature, syn_label
#
# # unseen_attributes, unseen_att_labels = load_unseen_att(args)
# # features, labels = generate_syn_feature(unseen_att_labels, unseen_attributes, args.syn_num)
#
# seen_attributes_with_bg, seen_att_labels_with_bg = load_seen_att(args)
# seen_attributes = seen_attributes_with_bg[1:,:]
# seen_att_labels = seen_att_labels_with_bg[1:]
#
# features, labels = generate_syn_feature(seen_att_labels, seen_attributes, args.syn_num)
#
#
# np.save(f'{args.save_dir}/{args.data_split}_feats.npy', features)
# np.save(f'{args.save_dir}/{args.data_split}_labels.npy', labels)

