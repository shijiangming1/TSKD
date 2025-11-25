from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from loss import TripletLoss_WRT, TripletLoss_ADP, RobustTripletLoss_final, MMD_loss,OriTripletLoss
from ChannelAug import ChannelAdapGray, ChannelRandomErasing
from sklearn.mixture import GaussianMixture
import datetime
import tarfile
def gen_code_archive(out_dir, file='code.tar.gz'):
    archive = os.path.join(out_dir, file)
    print(f"code save in {archive}")
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    with tarfile.open(archive, mode='w:gz') as tar:
        tar.add('.', filter=is_source_file)
    return archive

def is_source_file(x):
    if x.isdir() or x.name.endswith(('.py', '.sh', '.yml', '.json', '.txt', '.md')) \
            and '.mim' not in x.name and 'jobs/' not in x.name:
        # print(x.name)
        return x
    else:
        return None


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume-net1', default='', type=str,
                    help='resume net1 from checkpoint')
parser.add_argument('--model_path', default='./save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=1, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='robust', type=str,
                    metavar='m', help='method type: base or agw or robust')
parser.add_argument('--loss1', default='sid', type=str, help='loss type: id or soft id')
parser.add_argument('--loss2', default='adp', type=str,
                    metavar='m', help='loss type: wrt or adp or robust_tri')
parser.add_argument('--margin', default=0.2, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='6', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--savename', default='abalation_noise_0.00_SID_1_MD_0.05_SM_0.5_tri_batch_6_PN_4_384*192', type=str,
                    help='name of the saved model')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--augc', default=1, type=int,
                    metavar='aug', help='use channel aug or not')
parser.add_argument('--rande', default=0.5, type=float,
                    metavar='ra', help='use random erasing or not and the probability')
parser.add_argument('--kl', default=0, type=float,
                    metavar='kl', help='use kl loss and the weight')
parser.add_argument('--alpha', default=1, type=int,
                    metavar='alpha', help='magnification for the hard mining')
parser.add_argument('--gamma', default=1, type=int,
                    metavar='gamma', help='gamma for the hard mining')
parser.add_argument('--square', default=1, type=int,
                    metavar='square', help='gamma for the hard mining')
parser.add_argument('--noise-mode', default='sym', type=str, help='sym')
parser.add_argument('--noise-rate', default=0.00, type=float,
                    metavar='nr', help='noise_rate')
# parser.add_argument('--data-path', default='/data1/SYSU-MM01/', type=str, help='path to dataset')
parser.add_argument('--data-path', default='../../SYSU-MM01/', type=str, help='path to dataset')

parser.add_argument('--p-threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--warm-epoch', default=1, type=int, help='epochs for net warming up')
parser.add_argument('--UseCL', default=0, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--scale', default=2, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--sm_w', default=0.5, type=float,
                    metavar='t', help='the weight of self-mimic loss')
parser.add_argument('--md_w', default=0.05, type=float,
                    metavar='t', help='the weight of mutual distillation loss')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
data_path = args.data_path

if dataset == 'sysu':
    test_mode = [1, 2]  # thermal to visible
    log_path = args.log_path + 'sysu_log_ddag/'

elif dataset == 'regdb':
    test_mode = [2, 1]  # visible to thermal
    log_path = args.log_path + 'regdb_log_ddag/'
def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)

timestamp = time_str()

checkpoint_path = args.model_path

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
# test_log_file = open(log_path + args.savename + '.txt', "w")
sys.stdout = Logger(log_path + args.savename+timestamp + '.txt')
gen_code_archive(out_dir=os.path.join('results/', ), file=f'{ args.savename+timestamp}.tar.gz')

with open('./main.py', encoding='utf-8') as file:
    content = file.read()
    print(content.rstrip())
file.close() # 关闭打开的文件
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

if args.rande > 0:
    transform_train_list = transform_train_list + [ChannelRandomErasing(probability=args.rande)]

if args.augc == 1:
    transform_train_list = transform_train_list + [ChannelAdapGray(probability=0.5)]

transform_train = transforms.Compose(transform_train_list)

end = time.time()
if dataset == 'sysu':
    # evaltrain set
    evaltrainset = SYSUData(data_path, transform=transform_test, noise_rate=args.noise_rate,
                            noise_file='%s/%.2f_%s' % (args.data_path, args.noise_rate, args.noise_mode),
                            mode='evaltrain')
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(evaltrainset.train_color_label, evaltrainset.train_thermal_label)
    # create_pesudoset = SYSUData(data_path, transform=transform_train, noise_rate=args.noise_rate, mode='create_pesudo',
    #                             noise_file='%s/%.2f_%s' % (data_path, args.noise_rate, args.noise_mode))
    # pre_trainset = SYSUData(data_path,  transform=transform_train, noise_rate=args.noise_rate,
    #                       noise_file='%s/%.2f_%s' % (data_path, args.noise_rate, args.noise_mode),
    #                       mode='pre_train')
    create_pesudoset=0
    pre_trainset=0
    warmupset = SYSUData(data_path, transform=transform_train, noise_rate=args.noise_rate, mode='warmup',
                         noise_file='%s/%.2f_%s' % (args.data_path, args.noise_rate, args.noise_mode))

    trainset = SYSUData(data_path, transform=transform_train, noise_rate=args.noise_rate, mode='train',
                        noise_file='%s/%.2f_%s' % (args.data_path, args.noise_rate, args.noise_mode))

    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    evaltrainset = RegDBData(data_path,
                             trial=args.trial,
                             transform=transform_test,
                             noise_rate=args.noise_rate,
                             noise_file='%s/%.2f_%s' % (args.data_path, args.noise_rate, args.noise_mode),
                             mode='evaltrain')
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(evaltrainset.train_color_label, evaltrainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(evaltrainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(evaltrainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(evaltrainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method == 'base':
    net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)
elif args.method == 'adp':
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
elif args.method == 'robust':
    net1 = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
    net1.to(device)

pretrainnet = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
pretrainnet.to(device)
cudnn.benchmark = True

# define loss function
criterion_id = nn.CrossEntropyLoss()
criterion_CE = nn.CrossEntropyLoss(reduction='none')

loader_batch = args.batch_size * args.num_pos
criterion2 = OriTripletLoss(batch_size=loader_batch, margin=args.margin)

if args.loss2 == 'wrt':
    criterion_tri = TripletLoss_WRT()
    criterion_tri.to(device)
elif args.loss2 == 'adp':
    criterion_tri = TripletLoss_ADP(alpha=args.alpha, gamma=args.gamma, square=args.square)
    criterion_tri.to(device)
elif args.loss2 == 'robust_tri':
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = RobustTripletLoss_final(batch_size=loader_batch, margin=args.margin)
criterion_l1 = nn.L1Loss()
MMDLoss = MMD_loss().to(device)
# initial the prototype features
RGB_tensor1 = torch.zeros(n_class, 2048).cuda()
RGB_tensor2 = torch.zeros(n_class, 2048).cuda()
IR_tensor = torch.zeros(n_class, 2048).cuda()
RGB_SM_ALL_list = []
IR_SM_ALL_list = []
RGB_MD_ALL_list = []
IR_MD_ALL_list = []

if args.optim == 'sgd':
    ignored_params1 = list(map(id, net1.bottleneck.parameters())) \
                      + list(map(id, net1.classifier.parameters()))
    base_params1 = filter(lambda p: id(p) not in ignored_params1, net1.parameters())
    optimizer1 = optim.SGD([
        {'params': base_params1, 'lr': 0.1 * args.lr},
        {'params': net1.bottleneck.parameters(), 'lr': args.lr},
        {'params': net1.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)


if len(args.resume_net1) > 0:
    model_path1 = checkpoint_path + args.resume_net1
    if os.path.isfile(model_path1):
        print('==> loading checkpoint {}'.format(args.resume_net1))
        checkpoint1 = torch.load(model_path1)
        net1.load_state_dict(checkpoint1['net'])
        start_epoch = checkpoint1['epoch'] + 1
        optimizer1.load_state_dict(checkpoint1['optimizer'])

    else:
        print('==> no checkpoint found at {} '.format(args.resume_net1))

# initial the prototype features

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 40:
        lr = args.lr * 0.1
    elif epoch >= 40:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr

def pre_train(epoch, net1,net2, optimizer, dataloader):
    current_lr = adjust_learning_rate(optimizer, epoch)
    net1.train()
    net2.train()

    num_iter = (len(dataloader.dataset.cIndex) // dataloader.batch_size) + 1
    # epochs_pre=30
    for epoch_pre in range(epoch):
        for batch_idx, (input10, input11, input2, label1, label2) in enumerate(dataloader):
            n=input10.size(0)
            labels1 = torch.cat((label1, label1), 0)
            labels2 = torch.cat((label2, label2), 0)
            # input1 = torch.cat((input10, input11,), 0)
            input10 = input10.cuda()
            input11 = input11.cuda()

            input2 = input2.cuda()
            labels1 = labels1.cuda()
            labels1=labels1.long()
            labels2 = labels2.cuda()
            labels2=labels2.long()
            _, out0, = net1(input10, input11)
            _, out1, = net2(input2, input2)

            loss_id0 = criterion_id(out0, labels1)
            loss_id1 = criterion_id(out1, labels2)

            optimizer.zero_grad()
            loss_id0.backward()
            loss_id1.backward()

            optimizer.step()

            # optimizer.zero_grad()
            # loss_id1.backward()
            # optimizer.step()

            predict0 = out0.argmax(dim=1)
            predict1 = out1.argmax(dim=1)
            res_rgb = labels1 - predict0
            res_rgb = np.array(res_rgb.cpu())
            res_rgb = np.sum(res_rgb == 0)
            acc_rgb=res_rgb/(2*n)

            res_ir = labels2 - predict1
            res_ir = np.array(res_ir.cpu())
            res_ir = np.sum(res_ir == 0)
            acc_ir = res_ir / (2*n)

            # res_rgb = np.sum(res[:2*n] == 0)
            # acc_rgb = res_rgb / (2*n)
            # acc_rgb = 0



            if batch_idx % 50 == 0:
                print('%s:%.2f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f\t Current-lr: %.4f \t Current-rgb-acc: %.4f \t Current-ir-acc: %.4f'
                      % (args.dataset, args.noise_rate, args.noise_mode, epoch_pre, 10, batch_idx + 1,
                         num_iter, loss_id1.item(), current_lr,acc_rgb,acc_ir))
        state = {
            'net': net1.state_dict(),
            'epoch': epoch_pre,
        }
        # if epoch_pre%10==0:
        #     if args.dataset == 'sysu':
        #         torch.save(state, checkpoint_path +"pre_train/"+ "rate90V1.2" + "_epoch_{}_".format(epoch_pre)+'pre_net.t')
        #     else:
        #         torch.save(state, checkpoint_path + args.savename + 'pre_trial{}'.format(args.trial) +
        #                    '_net.t')
    return net1

def warmup(epoch, net, optimizer, dataloader):
    current_lr = adjust_learning_rate(optimizer, epoch)
    net.train()

    num_iter = (len(dataloader.dataset.cIndex) // dataloader.batch_size) + 1
    for batch_idx, (input10, input11, input2, label1, label2) in enumerate(dataloader):
        labels = torch.cat((label1, label1, label2), 0)
        input1 = torch.cat((input10, input11,), 0)

        input1 = input1.cuda()
        input2 = input2.cuda()
        labels = labels.cuda()

        _, out0, = net(input1, input2)
        loss_id = criterion_id(out0, labels)

        optimizer.zero_grad()
        loss_id.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f\t Current-lr: %.4f'
                  % (args.dataset, args.noise_rate, args.noise_mode, epoch, 80, batch_idx + 1,
                     num_iter, loss_id.item(), current_lr))

def eval_train(net, dataloader, type):
    losses_V_aug1 = -1. * torch.ones(len(evaltrainset.train_color_label))
    losses_V_aug2 = -1. * torch.ones(len(evaltrainset.train_color_label))
    losses_I = -1. * torch.ones(len(evaltrainset.train_thermal_label))

    net.train()
    with torch.no_grad():
        for batch_idx, (input10, input11, input2, label1, label2, index_V, index_I) in enumerate(dataloader):
            input1 = torch.cat((input10, input11,), 0)
            input1 = input1.cuda()
            input2 = input2.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()

            index_V = np.concatenate((index_V, index_V), 0)
            labels = torch.cat((label1, label1, label2), 0)
            _, out0, = net(input1, input2)
            loss = criterion_CE(out0, labels)
            loss1 = loss[0:input2.size(0)*2]
            loss2 = loss[input2.size(0)*2:input2.size(0)*3]

            for n1 in range(input2.size(0)):
                losses_V_aug1[index_V[n1]] = loss1[n1]
                losses_V_aug2[index_V[n1 + input2.size(0)]] = loss1[n1 + input2.size(0)]
            for n2 in range(input2.size(0)):
                losses_I[index_I[n2]] = loss2[n2]

    losses_V_aug1_slt = (losses_V_aug1 - losses_V_aug1.min()) / (losses_V_aug1.max() - losses_V_aug1.min())
    losses_V_aug2_slt = (losses_V_aug2 - losses_V_aug2.min()) / (losses_V_aug2.max() - losses_V_aug2.min())
    losses_I_slt = (losses_I - losses_I.min()) / (losses_I.max() - losses_I.min())
    losses_V_slt = torch.cat((losses_V_aug1_slt, losses_V_aug2_slt), 0)

    input_loss_V = losses_V_slt.reshape(-1, 1)
    input_loss_I = losses_I_slt.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm_V = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    gmm_V.fit(input_loss_V)
    prob_V = gmm_V.predict_proba(input_loss_V)
    prob_V = prob_V[:, gmm_V.means_.argmin()]

    gmm_I = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    gmm_I.fit(input_loss_I)
    prob_I = gmm_I.predict_proba(input_loss_I)
    prob_I = prob_I[:, gmm_I.means_.argmin()]

    return prob_V, prob_I

def create_pesudo(net, dataloader,noise,dataset):
    net.train()
    with torch.no_grad():
        acc_irs=0
        acc_rgbs=0
        for batch_idx, (input1, input2, label1, label2, cindex, tindex) in enumerate(dataloader):
            n=int(input1.size(0))
            labels = torch.cat((label1, label2), 0)
            input1 = input1.cuda()
            input2 = input2.cuda()
            labels = labels.cuda()
            _, out0 = net1(input1, input2)
            predict = out0.argmax(dim=1)
            res = labels - predict
            res = np.array(res.cpu())
            res_rgb = np.sum(res[0:n] == 0)
            res_ir = np.sum(res[n:] == 0)
            acc_rgb = res_rgb / n
            acc_ir = res_ir / n
            acc_irs+=acc_ir
            acc_rgbs+=acc_rgb

            warmupset.train_color_label[cindex] = predict[0:n].cpu()
            warmupset.train_thermal_label[tindex] = predict[n:].cpu()
            # evaltrainset.train_color_label[cindex] = predict[0:n].cpu()
            # evaltrainset.train_thermal_label[tindex] = predict[n:].cpu()
            # trainset.train_color_label[cindex] = predict[0:n].cpu()
            # trainset.train_thermal_label[tindex] = predict[n:].cpu()
        rgb_predictlabels=warmupset.train_color_label
        rgb_turelabels=warmupset.true_train_color_label
        res = rgb_predictlabels - rgb_turelabels
        res = np.array(res)
        res_rgb = np.sum(res == 0)
        acc_rgb = res_rgb / len(rgb_predictlabels)
        print("acc_rgb:",acc_rgb)
        ir_predictlabels = warmupset.train_thermal_label
        ir_turelabels = warmupset.true_train_thermal_label
        res = ir_predictlabels - ir_turelabels
        res = np.array(res)
        ir_rgb = np.sum(res == 0)
        acc_ir = ir_rgb / len(ir_predictlabels)
        print("acc_ir", acc_ir)
        # if dataset=="sysu":
        #     np.save(('./save_model/precess/' + 'rgb_P_whole_label_rate_{}.npy'.format(noise)), rgb_predictlabels)
        #     np.save(('./save_model/precess/' + 'ir_P_whole_label_rate_{}.npy'.format(noise)), ir_predictlabels)
        # else:
        #     np.save(('/data1/RegDB/' + 'rgb_P_adam_label_rate_{}.npy'.format(noise)), rgb_predictlabels)
        #     np.save(('/data1/RegDB/' + 'ir_P_adam_label_rate_{}.npy'.format(noise)), ir_predictlabels)
        trainset.train_color_label=rgb_predictlabels
        trainset.train_thermal_label=ir_predictlabels
        evaltrainset.train_color_label = rgb_predictlabels
        evaltrainset.train_thermal_label= ir_predictlabels
def train(epoch, net, optimizer, trainloader):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    rgb_SM_loss = AverageMeter()
    ir_SM_loss = AverageMeter()
    rgb_MD_loss = AverageMeter()
    ir_MD_loss = AverageMeter()
    cnt_sum = 0
    id_correct = 0
    tri_correct = 0
    total = 0

    net.train()
    end = time.time()
    for batch_idx, (input10, input11, input2, label1, label2, true_label1, true_labels2, probV_1, probV_2,
                    probI) in enumerate(trainloader):
        labels = torch.cat((label1, label1, label2), 0)
        input1 = torch.cat((input10, input11,), 0)
        true_labels = torch.cat((true_label1, true_label1, true_labels2), 0)
        probV_1 = probV_1.cuda()
        probV_2 = probV_2.cuda()
        probI = probI.cuda()
        prob = torch.cat((probV_1, probV_2, probI), 0)
        input1 = input1.cuda()
        input2 = input2.cuda()
        labels = labels.cuda()
        prob = prob.cuda()
        data_time.update(time.time() - end)
        loss_total = 0
        loss_SM_rgb = torch.zeros(1).cuda()
        loss_SM_ir = torch.zeros(1).cuda()
        feat, out0 = net(input1, input2)
        n = int(feat.size(0) / 3)
        feat_rgb1 = feat[0:n, :]
        feat_rgb2 = feat[n:2 * n, :]
        feat_ir = feat[2 * n:, :]
        if args.loss1 == 'sid':
            loss_id = criterion_CE(out0, labels)
            loss_id = prob * loss_id
            loss_id = loss_id.sum() / prob.size(0)
        else:
            loss_id = criterion_id(out0, labels)
        # total_loss+=loss_id

        # calculate self-mimic (SM) loss --- epoch: 1
        if epoch >= 1:
            for i, f in enumerate(feat_rgb1):
                rgb_id_index = label1[i]
                ir_id_index = label2[i]
                loss_SM_rgb +=( criterion_l1(feat_rgb1[i], RGB_tensor1[rgb_id_index].detach())*probV_1[i]+
                               criterion_l1(feat_rgb2[i], RGB_tensor2[rgb_id_index].detach())*probV_2[i])/2.
                loss_SM_ir += criterion_l1(feat_ir[i], IR_tensor[ir_id_index].detach())*probI[i]
            loss_SM_rgb = loss_SM_rgb * args.sm_w / feat_rgb1.shape[0]
            loss_SM_ir = loss_SM_ir * args.sm_w / feat_rgb1.shape[0]
            loss_total = loss_total + (loss_SM_rgb + loss_SM_ir)
            # print(loss_SM_rgb)
            # print(loss_SM_ir)

        # calculate mutual-distillation (MD) loss --- epoch: 10
        loss_MD_rgb1 = torch.zeros(1).cuda()
        loss_MD_rgb2 = torch.zeros(1).cuda()

        loss_MD_ir1 = torch.zeros(1).cuda()
        loss_MD_ir2 = torch.zeros(1).cuda()
        loss_MD_rgb = torch.zeros(1).cuda()
        loss_MD_ir = torch.zeros(1).cuda()

        if epoch >= 15:
            num_class = torch.unique(label1)
            for classi in range(len(num_class)):
                loss_MD_rgb1 += MMDLoss(feat_rgb1[label1 == num_class[classi]],
                                       feat_ir[label2 == num_class[classi]].detach())
                loss_MD_rgb2 += MMDLoss(feat_rgb2[label1 == num_class[classi]],
                                        feat_ir[label2 == num_class[classi]].detach())
                loss_MD_ir1 += MMDLoss(feat_ir[label2 == num_class[classi]],
                                      feat_rgb1[label1 == num_class[classi]].detach())
                loss_MD_ir2 += MMDLoss(feat_ir[label2 == num_class[classi]],
                                      feat_rgb2[label1 == num_class[classi]].detach())
            #
            loss_MD_ir = ((loss_MD_ir1+loss_MD_ir2) / len(num_class))*args.md_w
            loss_MD_rgb = ((loss_MD_rgb1+loss_MD_rgb2) / len(num_class))*args.md_w

            loss_total = loss_total + (loss_MD_ir + loss_MD_rgb)
        loss_tri, batch_acc, cnt = criterion_tri(feat, out0, labels, true_labels, prob, threshold=0.5)
        loss_tri, batch_acc = criterion2(feat, labels)
        loss_total=loss_total+loss_id+loss_tri
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # update
        train_loss.update(loss_total.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        total += labels.size(0)
        cnt_sum += int(cnt)
        tri_correct += batch_acc
        _, predicted = out0.max(1)
        id_correct += predicted.eq(labels).sum().item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx== 0:
            # RGB_SM_ALL_list
            RGB_SM_ALL_list.append(loss_SM_rgb.detach().cpu().numpy())
            IR_SM_ALL_list.append(loss_SM_ir.detach().cpu().numpy())
            RGB_MD_ALL_list.append(loss_MD_rgb.detach().cpu().numpy())
            IR_MD_ALL_list.append(loss_MD_ir.detach().cpu().numpy())
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.avg:.3f} '
                  'lr:{:.3f} '
                  'Loss: {train_loss.avg:.4f} '
                  'iLoss: {id_loss.avg:.4f} '
                  'TLoss: {tri_loss.avg:.4f} '
                  'rgb_SM: {rgb_SM_loss.val:.4f} ({rgb_SM_loss.avg:.4f}) '
                  'ir_SM: {ir_SM_loss.val:.4f} ({ir_SM_loss.avg:.4f}) '
                  'rgb_MD: {rgb_MD_loss.val:.4f} ({rgb_MD_loss.avg:.4f}) '
                  'ir_MD: {ir_MD_loss.val:.4f} ({ir_MD_loss.avg:.4f}) '
                  'id-acc: {:.2f} '
                  'tri-acc: {:.2f} '.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * id_correct / total, 100. * tri_correct / cnt_sum,
                batch_time=batch_time, train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss,
                rgb_SM_loss=rgb_SM_loss,
                ir_SM_loss=ir_SM_loss, rgb_MD_loss=rgb_MD_loss, ir_MD_loss=ir_MD_loss))
    # compute mean feature
    RGB_tensor_tmp1 = torch.zeros(n_class, 2048).cuda()
    RGB_tensor_tmp2 = torch.zeros(n_class, 2048).cuda()
    IR_tensor_tmp = torch.zeros(n_class, 2048).cuda()
    rgb_cnt_tmp = torch.zeros(n_class).cuda()
    ir_cnt_tmp = torch.zeros(n_class).cuda()

    # update the prototype of each person ID
    if epoch > -1:
        for batch_idx, (input10, input11, input2, label1, label2, true_label1, true_labels2, probV_1, probV_2,
                        probI) in enumerate(trainloader):
            labels = torch.cat((label1, label1, label2), 0)
            input1 = torch.cat((input10, input11,), 0)
            true_labels = torch.cat((true_label1, true_label1, true_labels2), 0)
            prob = torch.cat((probV_1, probV_2, probI), 0)
            probV_1=probV_1.cuda()
            probV_2 = probV_2.cuda()
            probI=probI.cuda()
            input1 = input1.cuda()
            input2 = input2.cuda()
            labels = labels.cuda()
            prob = prob.cuda()
            with torch.no_grad():
                feat, out0 = net(input1, input2)
                n = int(feat.size(0) / 3)
                feat_rgb1 = feat[0:n, :]*probV_1[:, None]
                feat_rgb2 = feat[n:2*n, :]*probV_2[:, None]
                feat_ir = feat[2*n:, :]*probI[:, None]
                for i, f in enumerate(feat_rgb1):
                    rgb_id_index = label1[i]
                    ir_id_index = label2[i]
                    RGB_tensor_tmp1[rgb_id_index] += feat_rgb1[i]
                    RGB_tensor_tmp2[rgb_id_index] += feat_rgb2[i]
                    IR_tensor_tmp[ir_id_index] += feat_ir[i]
                    rgb_cnt_tmp[rgb_id_index] += 1
                    ir_cnt_tmp[ir_id_index] += 1
        for i in range(n_class):
            if rgb_cnt_tmp[i] > 0:
                RGB_tensor1[i] = RGB_tensor_tmp1[i] / rgb_cnt_tmp[i]
                RGB_tensor2[i] = RGB_tensor_tmp2[i] / rgb_cnt_tmp[i]
            if ir_cnt_tmp[i] > 0:
                IR_tensor[i] = IR_tensor_tmp[i] / ir_cnt_tmp[i]


    return 1. / (1. + train_loss.avg)

def test(net1):
    # switch to evaluation mode
    net1.eval()
    ptr = 0
    gall_feat_att = np.zeros((ngall, 2048))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_att1 = net1(input, input, test_mode[0])
            feat_att =feat_att1
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num

    # switch to evaluation
    net1.eval()
    ptr = 0
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_att1 = net1(input, input, test_mode[1])
            feat_att = feat_att1
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num

    # compute the similarity
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)

    return cmc_att, mAP_att, mINP_att


# training
# if args.dataset == 'sysu':
#     warmupset = SYSUData(data_path, transform=transform_train, noise_rate=args.noise_rate, mode='warmup',
#                          noise_file='%s/%.1f_%s' % (args.data_path, args.noise_rate, args.noise_mode))
#     trainset = SYSUData(data_path, transform=transform_train, noise_rate=args.noise_rate, mode='train',
#                     noise_file='%s/%.1f_%s' % (args.data_path, args.noise_rate, args.noise_mode))
# elif args.dataset == 'regdb':
#     warmupset = RegDBData(data_path, trial=args.trial, transform=transform_train, noise_rate=args.noise_rate,
#                           noise_file='%s/%.1f_%s' % (args.data_path, args.noise_rate, args.noise_mode),
#                           mode='warmup')
#     trainset = RegDBData(data_path, trial=args.trial, transform=transform_train, noise_rate=args.noise_rate,
#                          noise_file='%s/%.1f_%s' % (args.data_path, args.noise_rate, args.noise_mode),
#                          mode='train')

if args.noise_rate>=0.0001:
    print('==> Preparing Data Loader...')
    loader_batch = args.batch_size * args.num_pos
    # model_path = '/data1/SSM/semi-DART04/save_model/pre_train/rate{:.2f}Vwhole_epoch_30_pre_net.t'.format(args.noise_rate)
    model_path=''
    suffix = dataset

    print('==> Start Pre_training...')

    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(model_path))
        checkpoint = torch.load(model_path)
        # start_epoch = checkpoint['epoch']
        start_epoch = 0

        # pdb.set_trace()
        net1.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(model_path, checkpoint['epoch']))
    else:
        pre_sampler = AllSampler(args.dataset, pre_trainset.train_color_label, pre_trainset.train_thermal_label)
        pre_trainset.cIndex = pre_sampler.index1  # color index
        pre_trainset.tIndex = pre_sampler.index2  # thermal index
        pre_trainloader = data.DataLoader(pre_trainset, batch_size=50, sampler=pre_sampler, \
                                          num_workers=args.workers, drop_last=False)
        net1 = pre_train(31, net1,net1, optimizer1, pre_trainloader)
    print('==> Finish Pre_training...')

    print('==> Start create_pesudo...')

    create_pesudo_label_sampler = AllSampler(args.dataset, create_pesudoset.train_color_label,
                                             create_pesudoset.train_thermal_label)
    create_pesudoset.cIndex = create_pesudo_label_sampler.index1  # color index
    create_pesudoset.tIndex = create_pesudo_label_sampler.index2  # thermal index
    create_pesudolabel_loader = data.DataLoader(create_pesudoset,
                                                batch_size=loader_batch,
                                                sampler=create_pesudo_label_sampler,
                                                num_workers=args.workers,
                                                drop_last=False)
    create_pesudo(net1, create_pesudolabel_loader,'%.2f' % (args.noise_rate),dataset)
    print('==> Finished create_pesudo...')
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

for epoch in range(start_epoch, 81):

    print('==> Preparing Data Loader...')
    loader_batch = args.batch_size * args.num_pos

    if epoch < args.warm_epoch:
        warmup_path1 = ''

        suffix = dataset
        if os.path.isfile(warmup_path1):
            print('==> loading checkpoint {}'.format(warmup_path1))
            checkpoint1 = torch.load(warmup_path1)
            # pdb.set_trace()
            net1.load_state_dict(checkpoint1['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(warmup_path1, checkpoint1['epoch']))
        else:
            warm_sampler = AllSampler(args.dataset, warmupset.train_color_label, warmupset.train_thermal_label)
            warmupset.cIndex = warm_sampler.index1  # color index
            warmupset.tIndex = warm_sampler.index2  # thermal index
            warmup_trainloader = data.DataLoader(warmupset, batch_size=loader_batch, sampler=warm_sampler, \
                                                 num_workers=args.workers, drop_last=True)
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader)
            print('\n')

    else:
        if epoch%1==0 and args.noise_rate>=0.0001:
            create_pesudo(net1, create_pesudolabel_loader, '%.2f' % (args.noise_rate), dataset)
            print('==> Finished create_pesudo...')
            color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        # models eval
        eval_sampler = AllSampler(args.dataset, evaltrainset.train_color_label, evaltrainset.train_thermal_label)

        evaltrainset.cIndex = eval_sampler.index1  # color index
        evaltrainset.tIndex = eval_sampler.index2  # thermal index

        eval_loader = data.DataLoader(evaltrainset,
                                      batch_size=loader_batch,
                                      sampler=eval_sampler,
                                      num_workers=args.workers,
                                      drop_last=True)

        prob_A_V, prob_A_I = eval_train(net1, eval_loader, 'A')

        TP = (prob_A_V[evaltrainset.rgb_cleanIdx] > args.p_threshold).sum() / len(evaltrainset.rgb_cleanIdx)
        TN = (prob_A_V[evaltrainset.rgb_noiseIdx] < args.p_threshold).sum() / len(evaltrainset.rgb_noiseIdx)
        FP = 1 - TN
        FN = 1 - TP

        print('Train Net1')
        trainset.probV_1, trainset.probV_2, trainset.probI = prob_A_V[0:int(len(prob_A_V) / 2)], prob_A_V[int(
            len(prob_A_V) / 2):], prob_A_I

        train_sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label,
                                        color_pos, thermal_pos, args.num_pos, args.batch_size, epoch)

        trainset.cIndex = train_sampler.index1  # color index
        trainset.tIndex = train_sampler.index2  # thermal index

        trainloader = data.DataLoader(
            dataset=trainset,
            batch_size=loader_batch,
            num_workers=args.workers,
            sampler=train_sampler,
            drop_last=True)

        # train net1
        train(epoch, net1, optimizer1, trainloader)


    if (epoch != 0) or epoch == 99:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc_att, mAP_att, mINP_att = test(net1)
        state = {
            'net': net1.state_dict(),
            'cmc': cmc_att,
            'mAP': mAP_att,
            'mINP': mINP_att,
            'epoch': epoch,
            'optimizer': optimizer1.state_dict()
        }

        # save model
        if (epoch >= args.warm_epoch ) or epoch == 99:
            # state = {
            #     'net': net1.state_dict(),
            #     'cmc': cmc_att,
            #     'mAP': mAP_att,
            #     'mINP': mINP_att,
            #     'epoch': epoch,
            #     'optimizer': optimizer1.state_dict()
            # }

            if args.dataset == 'sysu':
                # torch.save(state, checkpoint_path + args.savename+timestamp+'_epoch_{}'.format(epoch)+'_net1.t')
                torch.save(state, checkpoint_path + args.savename + timestamp + '_net1.t')
            else:
                torch.save(state, checkpoint_path + args.savename + '_trial{}'.format(args.trial) +
                           '_net1.t')


        print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'
              .format(cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))

        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            torch.save(state, checkpoint_path  + args.savename+timestamp+'_bestnet1.t')
            print('Best Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'
                  .format(cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
