import torch, math, time, argparse, os
import random, dataset, utils, losses, net
import numpy as np

from net.resnet import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from tqdm import *
import wandb

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

parser = argparse.ArgumentParser(description=
    'Official implementation of `IDML` on retrieval tasks' )
parser.add_argument('--LOG_DIR', 
    default='../logs',
    help = 'Path to log folder'
)
parser.add_argument('--dataset', 
    default='cub',
    help = 'Training dataset, e.g. cub, cars, SOP'
)
parser.add_argument('--embedding-size', default = 512, type = int,
    dest = 'sz_embedding',
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--batch-size', default = 150, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--epochs', default = 60, type = int,
    dest = 'nb_epochs',
    help = 'Number of training epochs.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 4, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--model', default = 'resnet50',
    help = 'Model for training'
)
parser.add_argument('--loss', default = 'Proxy_Anchor',
    help = 'Criterion for training'
)
parser.add_argument('--optimizer', default = 'adamw',
    help = 'Optimizer setting'
)
parser.add_argument('--lr', default = 1e-4, type =float,
    help = 'Learning rate setting'
)
parser.add_argument('--weight-decay', default = 1e-4, type =float,
    help = 'Weight decay setting'
)
parser.add_argument('--lr-decay-step', default = 10, type =int,
    help = 'Learning decay step setting'
)
parser.add_argument('--lr-decay-gamma', default = 0.5, type =float,
    help = 'Learning decay gamma setting'
)
parser.add_argument('--alpha', default = 32, type = float,
    help = 'Scaling Parameter setting'
)
parser.add_argument('--mrg', default = 0.1, type = float,
    help = 'Margin parameter setting'
)
parser.add_argument('--IPC', type = int,
    help = 'Balanced sampling, images per class'
)
parser.add_argument('--warm', default = 1, type = int,
    help = 'Warmup training epochs'
)
parser.add_argument('--bn-freeze', default = 1, type = int,
    help = 'Batch normalization parameter freeze'
)
parser.add_argument('--l2-norm', default = 1, type = int,
    help = 'L2 normlization'
)
parser.add_argument('--remark', default = '',
    help = 'Any reamrk'
)

args = parser.parse_args()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Directory for Log
LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(args.dataset, args.model, args.loss, args.sz_embedding, args.alpha, 
                                                                                            args.mrg, args.optimizer, args.lr, args.sz_batch, args.remark)
# Wandb Initialization
wandb.init(project=args.dataset + '_ProxyAnchor', notes=LOG_DIR)
wandb.config.update(args)

os.chdir('../data1/')
data_root = os.getcwd()
# Dataset Loader and Sampler
trn_dataset = dataset.load(
        name = args.dataset,
        root = data_root,
        mode = 'train',
        transform = dataset.utils.make_transform(
            is_train = True, 
            is_inception = (args.model == 'bn_inception')
        ))

if args.IPC:
    #上面返回一个列表，每个元素都是一个含IPC个相同类型图片下标的列表。下面是IPC*sz。检验是否正确：先在BA中的输出看是不是，再看第二排的输出是不是
    balanced_sampler = sampler.BASampler(trn_dataset, batch_size=args.sz_batch, images_per_class = args.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size = args.sz_batch, drop_last = True)
    # 以上给出索引，以下通过trn_dataset得到具体数据和标签
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = args.nb_workers,
        pin_memory = True,
        batch_sampler = batch_sampler
    )
    print('Balanced Sampling')
    
else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size = args.sz_batch,
        shuffle = True,
        num_workers = args.nb_workers,
        drop_last = True,
        pin_memory = True
    )
    print('Random Sampling')


ev_dataset = dataset.load(
        name = args.dataset,
        root = data_root,
        mode = 'eval',
        transform = dataset.utils.make_transform(
            is_train = False, 
            is_inception = (args.model == 'bn_inception')
        ))

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size = args.sz_batch,
    shuffle = False,
    num_workers = args.nb_workers,
    pin_memory = True
)

nb_classes = trn_dataset.nb_classes()

# Backbone Model
if args.model.find('googlenet')+1:
    model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
elif args.model.find('bn_inception')+1:
    model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet18')+1:
    model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet50')+1:
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet101')+1:
    model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)
    #即使没有 GPU，模型也可以以并行化的方式运行在 CPU 上，这有助于提高训练速度

# DML Losses
if args.loss == 'Proxy_Anchor':
    criterion = losses.Proxy_Anchor(nb_classes = nb_classes, sz_embed = args.sz_embedding, mrg = args.mrg, alpha = args.alpha).cuda()
elif args.loss == 'Proxy_NCA':
    criterion = losses.Proxy_NCA(nb_classes = nb_classes, sz_embed = args.sz_embedding).cuda()
elif args.loss == 'MS':
    criterion = losses.MultiSimilarityLoss().cuda()
elif args.loss == 'Contrastive':
    criterion = losses.ContrastiveLoss().cuda()
elif args.loss == 'Triplet':
    criterion = losses.TripletLoss().cuda()
elif args.loss == 'NPair':
    criterion = losses.NPairLoss().cuda()

# Train Parameters
param_groups = [
    {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu_id != -1 else 
                 list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
    {'params': model.model.embedding.parameters() if args.gpu_id != -1 else model.module.model.embedding.parameters(), 'lr':float(args.lr) * 1},
]
if args.loss == 'Proxy_Anchor':
    param_groups.append({'params': criterion.proxies, 'lr':float(args.lr) * 100})

def mixup(x, y, alpha):
    x=x.cuda()
    y=y.cuda()

    batch_size = x.size()[0]
    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(batch_size).cuda()

    mixed_x = lam*x + (1-lam)*x[index,:]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

# 在每个 epoch 结束时进行验证
def validate(model, dataloader):
    # model.eval()  # 将模型设置为评估模式
    # correct = [0] * 5  # 初始化 top-1 到 top-5 的正确预测数
    total = 0  # 记录总样本数
    accuracies = []
    # with torch.no_grad():  # 禁用梯度计算
        # for images, labels in dataloader:#验证集需要平衡采样吗？
        #     images = images.cuda()
        #     labels = labels.cuda()
            # outputs,_ = model(images)
            # _, predicted = outputs.topk(5, 1, largest=True, sorted=True)  # 获取前5个预测结果的索引
            #找到了相似度最大的索引
            # print('predict:',predicted)
            # print("length:",len(predicted))
    accuracies = utils.new_evaluate_cos(model, dl_ev)#TODO:从evaluate中改？？

            # total += labels.size(0)
            # for i in range(5):
            #     correct[i] += predicted[:, i].eq(labels).sum().item()  # 统计每个 top-k 的正确预测数
    # accuracy = [100 * c / total for c in correct]  # 计算每个 top-k 的准确率
    return accuracies

# Optimizer Setting
if args.optimizer == 'sgd': 
    opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay = args.weight_decay, momentum = 0.9, nesterov=True)
elif args.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
    
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma = args.lr_decay_gamma)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_recall=[0]
best_epoch = 0

for epoch in range(0, args.nb_epochs):
    model.train()
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    losses_per_epoch = []
    
    # Warmup: Train only new params, helps stabilize learning.
    if args.warm > 0:
        if args.gpu_id != -1:
            unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
        else:
            unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion.parameters())

        if epoch == 0:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == args.warm:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y) in pbar:      
        # if batch_idx == 3:
        #     break                   
        #x:bzxIPC y也是？？
        #x:150x3x224x224,y:150,m:150x512
        # print("x:",x)
        # print("x.shape:",x.shape)
        # print("y:",y)
        # print("y.shape:",y.shape)
        mixed_x, y_1, y_2, lam = mixup(x, y, 1.0)
        # mixup数据增强。y_b 是与 y 对应的另一组标签数据，它是根据原始标签数据 y 对应的混合后的输入数据 mixed_x 的索引 index 对应的标签数据
        m, v = model(x.squeeze().cuda())#默认是resnet50，只有这个模型进行了更改。分别输出embedding和uncertainty
        # print("m:",m)
        # print("m.shape:",m.shape)
        mixed_m, mixed_v = model(mixed_x.squeeze().cuda())
        loss_ori = criterion(m, v, y_1.squeeze().cuda())
        loss_mixed1 = criterion(mixed_m, mixed_v, y_1.squeeze().cuda())
        loss_mixed2 = criterion(mixed_m, mixed_v, y_2.squeeze().cuda())
        loss = loss_ori + lam*loss_mixed1 + (1-lam)*loss_mixed2
        
        opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        if args.loss == 'Proxy_Anchor':
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr),
                loss.item()))
        
    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({'loss': losses_list[-1]}, step=epoch)
    scheduler.step()
    
    top_k_accuracies = []  # 记录每个 epoch 的 top-k 准确率
    accuracies = validate(model, dl_ev)
    top_k_accuracies.append(accuracies)
    print(f"Epoch {epoch + 1}: Top-1 Accuracy: {accuracies[0]:.2f}%, Top-2 Accuracy: {accuracies[1]:.2f}%, Top-3 Accuracy: {accuracies[2]:.2f}%, Top-4 Accuracy: {accuracies[3]:.2f}%, Top-5 Accuracy: {accuracies[4]:.2f}%")

    # 定义存储模型的文件路径
    best_model_path = "best_model.pth"
    # 初始化最佳top-1准确率
    best_top1_accuracy = 0.0
    top1_accuracy = accuracies[0]
    # 如果当前 top-1 准确率大于历史最高准确率
    if top1_accuracy > best_top1_accuracy:
        # 更新最高准确率
        best_top1_accuracy = top1_accuracy
        # 保存当前模型
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch + 1}: Top-1 Accuracy improved to {top1_accuracy:.2f}%, Best model saved!")
    # 输出当前 epoch 的训练信息
    print(f"Epoch {epoch + 1}: Top-1 Accuracy: {top1_accuracy:.2f}%, Best Top-1 Accuracy: {best_top1_accuracy:.2f}%")

    """if(epoch >= 0):
        with torch.no_grad():
            print("**Evaluating...**")
            if args.dataset != 'SOP':
                F1, NMI, Recalls, MAP, RP = utils.evaluate_cos(model, dl_ev)
            else:
                F1, NMI, Recalls, MAP, RP = utils.evaluate_cos_SOP(model, dl_ev)
                
        # Logging Evaluation Score
        if args.dataset != 'SOP':
            for i in range(4):
                wandb.log({"R@{}".format(2**i): Recalls[i]}, step=epoch)
            wandb.log({"NMI":NMI}, step=epoch)
            wandb.log({"F1":F1}, step=epoch)
            wandb.log({"MAP":MAP}, step=epoch)
            wandb.log({"RP":RP}, step=epoch)
        else:
            for i in range(3):
                wandb.log({"R@{}".format(10**i): Recalls[i]}, step=epoch)
            wandb.log({"NMI":NMI}, step=epoch)
            wandb.log({"F1":F1}, step=epoch)
            wandb.log({"MAP":MAP}, step=epoch)
            wandb.log({"RP":RP}, step=epoch)
        
        # Best model save
        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch
            if not os.path.exists('{}'.format(LOG_DIR)):
                os.makedirs('{}'.format(LOG_DIR))
            torch.save({'model_state_dict':model.state_dict()}, '{}/{}_{}_best.pth'.format(LOG_DIR, args.dataset, args.model))
            with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
                f.write('Best Epoch: {}\n'.format(best_epoch))
                if args.dataset != 'SOP':
                    for i in range(4):
                        f.write("Best Recall@{}: {:.4f}\n".format(2**i, best_recall[i] * 100))
                else:
                    for i in range(3):
                        f.write("Best Recall@{}: {:.4f}\n".format(10**i, best_recall[i] * 100))
    
"""