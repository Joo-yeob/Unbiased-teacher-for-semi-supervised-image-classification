import argparse
import logging
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

# ============================================================
# Unbiased Teacher를 위한 추가/수정된 함수
# ============================================================

@torch.no_grad()
def update_teacher_model(student_model, teacher_model, ema_decay):
    """
    Student 모델의 가중치를 EMA(Exponential Moving Average)를 사용하여 Teacher 모델에 업데이트합니다.
    이 함수는 역전파를 사용하지 않고 안정적으로 Teacher를 업데이트하는 Unbiased Teacher의 핵심입니다.
    """
    student_params = student_model.state_dict()
    teacher_params = teacher_model.state_dict()
    
    for (k_s, v_s), (k_t, v_t) in zip(student_params.items(), teacher_params.items()):
        teacher_params[k_t] = ema_decay * v_t + (1. - ema_decay) * v_s
    teacher_model.load_state_dict(teacher_params)


logger = logging.getLogger(__name__)
best_acc = 0

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Unbiased Teacher Training')
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str, choices=['wideresnet', 'resnext'], help='architecture name')
    parser.add_argument('--total-steps', default=2**20, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int, help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate for teacher model')
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
    parser.add_argument('--out', default='result', help='directory to output the result')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")

    args = parser.parse_args()
    global best_acc

    # ============================================================
    # 모델 생성 함수
    # ============================================================
    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
        return model

    device = torch.device('cuda', args.gpu_id)
    args.device = device
    args.n_gpu = 1 # Single GPU training

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, './data')

    train_sampler = RandomSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # ============================================================
    # Unbiased Teacher: Student와 Teacher 모델을 별도로 생성
    # ============================================================
    student_model = create_model(args).to(args.device)
    teacher_model = create_model(args).to(args.device)
    
    # Teacher 모델의 파라미터를 Student와 동일하게 초기화하고, gradient 계산은 하지 않음
    teacher_model.load_state_dict(student_model.state_dict())
    for param in teacher_model.parameters():
        param.detach_() # gradient가 흐르지 않도록 함

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # Optimizer는 Student 모델만 업데이트
    optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov)

    #args.epochs = math.ceil(args.total_steps / args.eval_step)
    args.epochs = 100
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    student_model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          student_model, teacher_model, optimizer, scheduler)

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          student_model, teacher_model, optimizer, scheduler):
    global best_acc
    test_accs = []
    
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    student_model.train()
    teacher_model.train() # Dropout 등을 위해 train 모드로 설정 (하지만 역전파는 안 함)

    for epoch in range(args.start_epoch, args.epochs):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            inputs_x, targets_x = inputs_x.to(args.device), targets_x.to(args.device)
            inputs_u_w, inputs_u_s = inputs_u_w.to(args.device), inputs_u_s.to(args.device)

            # 1. Teacher로 Pseudo-Label 생성 (Weak Augmentation 데이터 사용)
            with torch.no_grad():
                teacher_logits = teacher_model(inputs_u_w)
                pseudo_probs = torch.softmax(teacher_logits / args.T, dim=-1)
                max_probs, pseudo_targets = torch.max(pseudo_probs, dim=-1)
                mask = (max_probs >= args.threshold).float()

            # 2. Student 학습
            # Supervised Loss (Labeled Data)
            student_logits_x = student_model(inputs_x)
            loss_s = F.cross_entropy(student_logits_x, targets_x, reduction='mean')
            
            # Unsupervised Loss (Unlabeled Data with Pseudo-Labels)
            student_logits_u = student_model(inputs_u_s)
            loss_u = (F.cross_entropy(student_logits_u, pseudo_targets, reduction='none') * mask).mean()

            # 3. 최종 손실 및 업데이트
            total_loss = loss_s + args.lambda_u * loss_u
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 4. Teacher 모델 EMA 업데이트
            update_teacher_model(student_model, teacher_model, args.ema_decay)

            losses.update(total_loss.item())
            losses_x.update(loss_s.item())
            losses_u.update(loss_u.item())
            mask_probs.update(mask.mean().item())
            
            if not args.no_progress:
                p_bar.set_description(f"Train Epoch: {epoch+1}/{args.epochs}. Iter: {batch_idx+1}/{args.eval_step}. LR: {scheduler.get_last_lr()[0]:.4f}. Loss: {losses.avg:.4f}. Loss_x: {losses_x.avg:.4f}. Loss_u: {losses_u.avg:.4f}. Mask: {mask_probs.avg:.2f}")
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        # 평가는 안정적인 Teacher 모델로 수행
        test_loss, test_acc = test(args, test_loader, teacher_model, epoch)

        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
        args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        # 저장 시 student와 teacher 모델 모두 저장
        model_to_save = student_model.module if hasattr(student_model, "module") else student_model
        teacher_to_save = teacher_model.module if hasattr(teacher_model, "module") else teacher_model
        
        save_checkpoint({
            'epoch': epoch + 1,
            'student_state_dict': model_to_save.state_dict(),
            'teacher_state_dict': teacher_to_save.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.out)

        test_accs.append(test_acc)
        logger.info(f'Best top-1 acc: {best_acc:.2f}')
        logger.info(f'Mean top-1 acc: {np.mean(test_accs[-20:]):.2f}\n')

    args.writer.close()

def test(args, test_loader, model, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if not args.no_progress:
        test_loader_tqdm = tqdm(test_loader)
    else:
        test_loader_tqdm = test_loader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader_tqdm):
            model.eval()
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            
            if not args.no_progress:
                 test_loader_tqdm.set_description(f"Test Iter: {batch_idx+1}/{len(test_loader)}. Loss: {losses.avg:.4f}. top1: {top1.avg:.2f}. top5: {top5.avg:.2f}.")

    logger.info(f"top-1 acc: {top1.avg:.2f}")
    logger.info(f"top-5 acc: {top5.avg:.2f}")
    return losses.avg, top1.avg

if __name__ == '__main__':
    main()
