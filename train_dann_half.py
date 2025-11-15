import datetime
import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)

# 可视化目标域分割结果

# 设置matplotlib后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal

ration = 3


# 新增：梯度反转层（GRL）实现
class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层，在前向传播中不变，反向传播中反转梯度"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# 新增：DANN域分类器
class DomainClassifier(torch.nn.Module):
    def __init__(self, input_channels=128, hidden_size=1024, num_domains=2):
        super(DomainClassifier, self).__init__()
        # 使用全局平均池化将特征图转换为特征向量
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_channels, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_size, num_domains),
        )

        # 初始化权重
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


import torch.nn.functional as F


# 修改：DANN版本的DeepLabV3+模型，添加多分类器
class DeepLabDANN(torch.nn.Module):
    def __init__(self, num_classes, backbone, downsample_factor, pretrained, lambda_domain=0.5):
        super(DeepLabDANN, self).__init__()
        # 原始DeepLabV3+模型
        self.deeplab = DeepLab(num_classes=num_classes, backbone=backbone,
                               downsample_factor=downsample_factor, pretrained=pretrained)

        # 获取backbone的输出通道数
        if backbone == 'ghostnet':
            backbone_output_channels = 96  # GhostNet最后一层的实际通道数

        # 第一个域分类器 - 在backbone后面
        self.domain_classifier_backbone = DomainClassifier(backbone_output_channels)

        # 第二个域分类器 - 在ASPP后面
        aspp_output_channels = 128  # ASPP输出特征通道数
        low_level_channels = 12  # ASPP输出特征通道数
        self.domain_classifier_aspp = DomainClassifier(low_level_channels)

        self.lambda_domain = lambda_domain  # 域对抗损失权重
        self.alpha = 0  # GRL参数，将在训练中动态调整

    def forward(self, x, alpha=None, mode='train'):
        if alpha is None:
            alpha = self.alpha

        # 获取输入图像尺寸
        H, W = x.size(2), x.size(3)

        # 获取DeepLabV3+的特征
        low_level_features_backbone, x_backbone = self.deeplab.backbone(x)

        # 继续分割解码过程
        x_aspp = self.deeplab.aspp(x_backbone)
        low_level_features = self.deeplab.shortcut_conv(low_level_features_backbone)
        x_aspp = F.interpolate(x_aspp, size=(low_level_features.size(2), low_level_features.size(3)),
                               mode='bilinear', align_corners=True)
        cls_conv_before = self.deeplab.cat_conv(torch.cat((x_aspp, low_level_features), dim=1))
        x = self.deeplab.cls_conv(cls_conv_before)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        # 如果是训练模式，计算两个域分类器的损失
        if mode == 'train':

            # 应用梯度反转层到ASPP特征
            reversed_features_aspp = GradientReversalLayer.apply(low_level_features_backbone, alpha)
            domain_output_aspp = self.domain_classifier_aspp(reversed_features_aspp)
            # 应用梯度反转层到backbone特征
            reversed_features_backbone = GradientReversalLayer.apply(x_backbone, alpha)
            domain_output_backbone = self.domain_classifier_backbone(reversed_features_backbone)

            return x, domain_output_backbone, domain_output_aspp

        else:
            # 验证/测试模式：直接返回分割结果
            return x

    def set_alpha(self, alpha):
        self.alpha = alpha


# 修改后的训练函数（半监督DANN版本）
def fit_one_epoch_semi_supervised_dann(model_train, model, loss_history, eval_callback, optimizer, epoch,
                                       epoch_step, epoch_step_val, gen_source, gen_target_labeled, gen_target_unlabeled,
                                       gen_val, UnFreeze_Epoch, Cuda,
                                       dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                                       save_period, save_dir, local_rank, lambda_domain=0.5, lambda_supervised=1.0):
    total_loss = 0
    total_seg_loss = 0
    total_supervised_target_loss = 0  # 新增：目标域有监督损失
    total_domain_loss_backbone = 0
    total_domain_loss_aspp = 0
    total_domain_loss = 0
    val_loss = 0

    # 新增：记录域分类准确率
    total_domain_acc_backbone_source = 0
    total_domain_acc_backbone_target = 0
    total_domain_acc_aspp_source = 0
    total_domain_acc_aspp_target = 0
    total_domain_acc = 0

    # 新增：记录分割准确率
    total_seg_acc = 0
    total_supervised_target_acc = 0  # 新增：目标域有监督准确率
    val_seg_acc = 0

    # 新增：记录每个类别的准确率
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    class_iou = np.zeros(num_classes)

    # 动态调整GRL参数（从0到1）
    p = float(epoch) / UnFreeze_Epoch
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    # 修复：通过module属性访问原始模型的set_alpha方法
    if hasattr(model_train, 'module'):
        model_train.module.set_alpha(alpha)  # 多GPU情况
    else:
        model_train.set_alpha(alpha)  # 单GPU情况

    if local_rank == 0:
        print('Start Semi-Supervised DANN Training')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}', postfix=dict, mininterval=0.3)

    # 设置模型为训练模式
    model_train.train()

    # 创建数据迭代器
    source_iter = iter(gen_source)
    target_labeled_iter = iter(gen_target_labeled)
    target_unlabeled_iter = iter(gen_target_unlabeled)

    for iteration in range(epoch_step):
        try:
            # 获取源域数据
            source_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(gen_source)
            source_batch = next(source_iter)

        try:
            # 获取有标注目标域数据
            target_labeled_batch = next(target_labeled_iter)
        except StopIteration:
            target_labeled_iter = iter(gen_target_labeled)
            target_labeled_batch = next(target_labeled_iter)

        try:
            # 获取无标注目标域数据
            target_unlabeled_batch = next(target_unlabeled_iter)
        except StopIteration:
            target_unlabeled_iter = iter(gen_target_unlabeled)
            target_unlabeled_batch = next(target_unlabeled_iter)

        # 处理源域数据
        imgs_source, pngs_source, labels_source = source_batch
        domain_labels_source = torch.zeros(imgs_source.size(0), dtype=torch.long)  # 源域标签为0

        # 处理有标注目标域数据
        imgs_target_labeled, pngs_target_labeled, labels_target_labeled = target_labeled_batch
        domain_labels_target_labeled = torch.ones(imgs_target_labeled.size(0), dtype=torch.long)  # 目标域标签为1

        # 处理无标注目标域数据
        imgs_target_unlabeled, _, _ = target_unlabeled_batch
        domain_labels_target_unlabeled = torch.ones(imgs_target_unlabeled.size(0), dtype=torch.long)  # 目标域标签为1

        if Cuda:
            imgs_source = imgs_source.cuda()
            pngs_source = pngs_source.cuda()
            domain_labels_source = domain_labels_source.cuda()

            imgs_target_labeled = imgs_target_labeled.cuda()
            pngs_target_labeled = pngs_target_labeled.cuda()
            domain_labels_target_labeled = domain_labels_target_labeled.cuda()

            imgs_target_unlabeled = imgs_target_unlabeled.cuda()
            domain_labels_target_unlabeled = domain_labels_target_unlabeled.cuda()

        with torch.cuda.amp.autocast(enabled=fp16):
            # 源域前向传播
            result_source = model_train(imgs_source, alpha=alpha, mode='train')
            seg_output_source, domain_output_backbone_source, domain_output_aspp_source = result_source

            # 有标注目标域前向传播
            result_target_labeled = model_train(imgs_target_labeled, alpha=alpha, mode='train')
            seg_output_target_labeled, domain_output_backbone_target_labeled, domain_output_aspp_target_labeled = result_target_labeled

            # 无标注目标域前向传播（只用于域分类）
            _, domain_output_backbone_target_unlabeled, domain_output_aspp_target_unlabeled = model_train(
                imgs_target_unlabeled, alpha=alpha, mode='train')

            # 计算分割损失（源域 + 有标注目标域）
            seg_loss_source = compute_segmentation_loss(seg_output_source, pngs_source, weights=cls_weights,
                                                        num_classes=num_classes, dice_loss=dice_loss,
                                                        focal_loss=focal_loss)

            # 有标注目标域的分割损失
            seg_loss_target_labeled = compute_segmentation_loss(seg_output_target_labeled, pngs_target_labeled,
                                                                weights=cls_weights, num_classes=num_classes,
                                                                dice_loss=dice_loss, focal_loss=focal_loss)

            # 总分割损失 = 源域分割损失 + λ * 有标注目标域分割损失
            seg_loss = seg_loss_source + lambda_supervised * seg_loss_target_labeled

            # 计算域分类损失（所有数据）
            # 源域域分类损失
            domain_loss_backbone_source = torch.nn.functional.cross_entropy(domain_output_backbone_source,
                                                                            domain_labels_source)
            domain_loss_aspp_source = torch.nn.functional.cross_entropy(domain_output_aspp_source, domain_labels_source)

            # 有标注目标域域分类损失
            domain_loss_backbone_target_labeled = torch.nn.functional.cross_entropy(
                domain_output_backbone_target_labeled, domain_labels_target_labeled)
            domain_loss_aspp_target_labeled = torch.nn.functional.cross_entropy(domain_output_aspp_target_labeled,
                                                                                domain_labels_target_labeled)

            # 无标注目标域域分类损失
            domain_loss_backbone_target_unlabeled = torch.nn.functional.cross_entropy(
                domain_output_backbone_target_unlabeled, domain_labels_target_unlabeled)
            domain_loss_aspp_target_unlabeled = torch.nn.functional.cross_entropy(domain_output_aspp_target_unlabeled,
                                                                                  domain_labels_target_unlabeled)

            # 总域分类损失（所有数据的平均）
            domain_loss_backbone = (
                                               domain_loss_backbone_source + domain_loss_backbone_target_labeled + domain_loss_backbone_target_unlabeled) / 3
            domain_loss_aspp = (
                                           domain_loss_aspp_source + domain_loss_aspp_target_labeled + domain_loss_aspp_target_unlabeled) / 3
            domain_loss = (domain_loss_backbone + domain_loss_aspp) / 2

            # 总损失 = 分割损失 + λ * 域对抗损失
            loss = seg_loss + lambda_domain * domain_loss

            # 新增：计算域分类准确率
            domain_pred_backbone_source = torch.argmax(domain_output_backbone_source, dim=1)
            domain_acc_backbone_source = (domain_pred_backbone_source == domain_labels_source).float().mean()

            domain_pred_backbone_target_labeled = torch.argmax(domain_output_backbone_target_labeled, dim=1)
            domain_acc_backbone_target_labeled = (
                        domain_pred_backbone_target_labeled == domain_labels_target_labeled).float().mean()

            domain_pred_backbone_target_unlabeled = torch.argmax(domain_output_backbone_target_unlabeled, dim=1)
            domain_acc_backbone_target_unlabeled = (
                        domain_pred_backbone_target_unlabeled == domain_labels_target_unlabeled).float().mean()

            domain_pred_aspp_source = torch.argmax(domain_output_aspp_source, dim=1)
            domain_acc_aspp_source = (domain_pred_aspp_source == domain_labels_source).float().mean()

            domain_pred_aspp_target_labeled = torch.argmax(domain_output_aspp_target_labeled, dim=1)
            domain_acc_aspp_target_labeled = (
                        domain_pred_aspp_target_labeled == domain_labels_target_labeled).float().mean()

            domain_pred_aspp_target_unlabeled = torch.argmax(domain_output_aspp_target_unlabeled, dim=1)
            domain_acc_aspp_target_unlabeled = (
                        domain_pred_aspp_target_unlabeled == domain_labels_target_unlabeled).float().mean()

            domain_acc = (
                                     domain_acc_backbone_source + domain_acc_backbone_target_labeled + domain_acc_backbone_target_unlabeled +
                                     domain_acc_aspp_source + domain_acc_aspp_target_labeled + domain_acc_aspp_target_unlabeled) / 6

            # 新增：计算分割准确率（源域 + 有标注目标域）
            seg_pred_source = torch.argmax(seg_output_source, dim=1)
            seg_acc_source = (seg_pred_source == pngs_source).float()
            mask_source = pngs_source != 255
            seg_acc_source = seg_acc_source[mask_source].mean() if mask_source.any() else torch.tensor(0.0).to(
                seg_acc_source.device)

            seg_pred_target_labeled = torch.argmax(seg_output_target_labeled, dim=1)
            seg_acc_target_labeled = (seg_pred_target_labeled == pngs_target_labeled).float()
            mask_target = pngs_target_labeled != 255
            seg_acc_target_labeled = seg_acc_target_labeled[mask_target].mean() if mask_target.any() else torch.tensor(
                0.0).to(seg_acc_target_labeled.device)

            seg_acc = (seg_acc_source + seg_acc_target_labeled) / 2

            # 新增：计算每个类别的准确率（源域 + 有标注目标域）
            for c in range(num_classes):
                # 源域
                mask_c_source = (pngs_source == c)
                if mask_c_source.any():
                    correct_c_source = (
                                seg_pred_source[mask_c_source] == pngs_source[mask_c_source]).float().sum().item()
                    class_correct[c] += correct_c_source
                    class_total[c] += mask_c_source.sum().item()

                    # 计算每个类别的IoU
                    pred_c_source = (seg_pred_source == c)
                    target_c_source = (pngs_source == c)

                    intersection = (pred_c_source & target_c_source).float().sum().item()
                    union = (pred_c_source | target_c_source).float().sum().item()

                    if union > 0:
                        class_iou[c] += intersection / union

                # 有标注目标域
                mask_c_target = (pngs_target_labeled == c)
                if mask_c_target.any():
                    correct_c_target = (seg_pred_target_labeled[mask_c_target] == pngs_target_labeled[
                        mask_c_target]).float().sum().item()
                    class_correct[c] += correct_c_target
                    class_total[c] += mask_c_target.sum().item()

                    # 计算每个类别的IoU
                    pred_c_target = (seg_pred_target_labeled == c)
                    target_c_target = (pngs_target_labeled == c)

                    intersection = (pred_c_target & target_c_target).float().sum().item()
                    union = (pred_c_target | target_c_target).float().sum().item()

                    if union > 0:
                        class_iou[c] += intersection / union

        # 反向传播
        optimizer.zero_grad()
        if fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_supervised_target_loss += seg_loss_target_labeled.item()
        total_domain_loss_backbone += domain_loss_backbone.item()
        total_domain_loss_aspp += domain_loss_aspp.item()
        total_domain_loss += domain_loss.item()

        # 新增：累加准确率
        total_domain_acc_backbone_source += domain_acc_backbone_source.item()
        total_domain_acc_backbone_target += (
                                                        domain_acc_backbone_target_labeled.item() + domain_acc_backbone_target_unlabeled.item()) / 2
        total_domain_acc_aspp_source += domain_acc_aspp_source.item()
        total_domain_acc_aspp_target += (
                                                    domain_acc_aspp_target_labeled.item() + domain_acc_aspp_target_unlabeled.item()) / 2
        total_domain_acc += domain_acc.item()
        total_seg_acc += seg_acc.item()
        total_supervised_target_acc += seg_acc_target_labeled.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'seg_loss': total_seg_loss / (iteration + 1),
                'sup_target_loss': total_supervised_target_loss / (iteration + 1),
                'domain_loss': total_domain_loss / (iteration + 1),
                'domain_acc': total_domain_acc / (iteration + 1),
                'seg_acc': total_seg_acc / (iteration + 1),
                'sup_target_acc': total_supervised_target_acc / (iteration + 1),
                'alpha': alpha,
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')

    # 验证过程
    model_train.eval()
    if local_rank == 0:
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}', postfix=dict, mininterval=0.3)

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            if Cuda:
                imgs = imgs.cuda()
                pngs = pngs.cuda()

            # 验证时只使用分割输出
            outputs = model_train(imgs, mode='val')
            loss = compute_segmentation_loss(outputs, pngs, weights=cls_weights,
                                             num_classes=num_classes, dice_loss=dice_loss,
                                             focal_loss=focal_loss)
            val_loss += loss.item()

            # 新增：计算验证集分割准确率
            seg_pred = torch.argmax(outputs, dim=1)
            seg_acc = (seg_pred == pngs).float()
            # 忽略标签为255的像素（如果有）
            mask = pngs != 255
            seg_acc = seg_acc[mask].mean() if mask.any() else torch.tensor(0.0).to(seg_acc.device)
            val_seg_acc += seg_acc.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'val_seg_acc': val_seg_acc / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        # 记录损失和准确率
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        # 新增：记录准确率
        loss_history.append_acc(epoch + 1,
                                total_seg_acc / epoch_step,
                                val_seg_acc / epoch_step_val,
                                total_domain_acc / epoch_step,
                                total_domain_acc_backbone_source / epoch_step,
                                total_domain_acc_backbone_target / epoch_step,
                                total_domain_acc_aspp_source / epoch_step,
                                total_domain_acc_aspp_target / epoch_step)

        print('Epoch:' + str(epoch + 1) + '/' + str(UnFreeze_Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        print('Supervised Target Loss: %.3f' % (total_supervised_target_loss / epoch_step))
        print('Domain Acc: %.3f%% (Backbone Source: %.3f%%, Target: %.3f%%)' %
              (total_domain_acc / epoch_step * 100,
               total_domain_acc_backbone_source / epoch_step * 100,
               total_domain_acc_backbone_target / epoch_step * 100))
        print('Domain Acc: %.3f%% (ASPP Source: %.3f%%, Target: %.3f%%)' %
              (total_domain_acc / epoch_step * 100,
               total_domain_acc_aspp_source / epoch_step * 100,
               total_domain_acc_aspp_target / epoch_step * 100))
        print('Seg Acc: %.3f%%, Sup Target Acc: %.3f%%' %
              (total_seg_acc / epoch_step * 100, total_supervised_target_acc / epoch_step * 100))

        # 打印每个类别的准确率和IoU
        print("\nClass-wise Performance:")
        for c in range(num_classes):
            if class_total[c] > 0:
                acc = class_correct[c] / class_total[c] * 100
                iou = class_iou[c] / epoch_step * 100
                print(f"Class {c}: Acc={acc:.2f}%, IoU={iou:.2f}%")
            else:
                print(f"Class {c}: No samples")

        # 保存模型
        if (epoch + 1) % save_period == 0 or epoch + 1 == UnFreeze_Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' %
                                                        (
                                                            epoch + 1, total_loss / epoch_step,
                                                            val_loss / epoch_step_val)))

        if eval_callback:
            eval_callback.on_epoch_end(epoch + 1, model_train)

    return total_loss / epoch_step, val_loss / epoch_step_val


# 辅助函数：计算分割损失（改进版，支持dice_loss和focal_loss）
def compute_segmentation_loss(outputs, targets, weights, num_classes, dice_loss, focal_loss):
    """
    计算分割损失，支持dice_loss和focal_loss
    """
    # 确保权重在正确的设备上
    weight_tensor = torch.from_numpy(weights).float()
    if torch.cuda.is_available():
        weight_tensor = weight_tensor.cuda()

    # 基础损失：带权重的交叉熵损失
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=255)
    ce_loss = ce_loss_fn(outputs, targets)

    total_loss = ce_loss

    # 添加Dice损失（对类别不平衡更鲁棒）
    if dice_loss:
        dice_loss_value = 0
        smooth = 1e-5
        outputs_soft = torch.softmax(outputs, dim=1)

        for c in range(num_classes):
            output_c = outputs_soft[:, c]
            target_c = (targets == c).float()

            intersection = (output_c * target_c).sum()
            union = output_c.sum() + target_c.sum()

            dice_loss_value += 1 - (2. * intersection + smooth) / (union + smooth)

        dice_loss_value /= num_classes
        total_loss += dice_loss_value

    # 添加Focal损失（关注难样本）
    if focal_loss:
        gamma = 2.0
        alpha = 0.25

        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        probs = torch.exp(log_probs)

        # 只考虑目标类别的概率
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)
        focal_weight = (1 - probs) ** gamma
        focal_weight = focal_weight * targets_one_hot

        # 应用类别权重
        weight_tensor = weight_tensor.view(1, num_classes, 1, 1)
        focal_weight = focal_weight * weight_tensor

        # 计算Focal损失
        focal_loss_value = -alpha * focal_weight * log_probs
        focal_loss_value = focal_loss_value.sum() / (targets.numel() + 1e-8)

        total_loss += focal_loss_value

    return total_loss


# 辅助函数：获取当前学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 计算类别权重函数
def calculate_class_weights(lines, VOCdevkit_path, num_classes):
    """
    计算每个类别的像素比例
    """
    class_counts = np.zeros(num_classes, dtype=np.float32)
    total_pixels = 0

    for line in tqdm(lines, desc="Calculating class weights"):
        name = line.split()[0]
        label_path = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass", name + ".png")
        label = np.array(Image.open(label_path))

        # 忽略标签为255的像素（如果有）
        mask = label != 255
        label = label[mask]

        for c in range(num_classes):
            class_counts[c] += np.sum(label == c)

        total_pixels += label.size

    class_ratios = class_counts / total_pixels
    print("\nClass Distribution:")
    for c in range(num_classes):
        print(f"Class {c}: {class_counts[c]} pixels ({class_ratios[c] * 100:.2f}%)")

    return class_ratios


# 计算类别权重
def compute_class_weights(class_ratios, method='median_frequency'):
    """
    根据类别比例计算类别权重

    参数:
    class_ratios: 每个类别的像素比例
    method: 权重计算方法 ('inverse', 'median_frequency', 'log', 'custom')

    返回:
    cls_weights: 计算出的类别权重
    """
    if method == 'inverse':
        # 方法1：使用比例倒数
        cls_weights = 1.0 / (class_ratios + 1e-8)  # 添加小值避免除以0
        cls_weights = cls_weights / cls_weights.min()  # 归一化，使最小权重为1

    elif method == 'median_frequency':
        # 方法2：使用中位数频率平衡
        median_frequency = np.median(class_ratios)
        cls_weights = median_frequency / (class_ratios + 1e-8)
        cls_weights = cls_weights / cls_weights.min()  # 归一化

    elif method == 'log':
        # 方法3：对数平衡（处理极端不平衡）
        cls_weights = np.log(1.02 / (class_ratios + 0.02))
        cls_weights = cls_weights / cls_weights.min()  # 归一化

    elif method == 'custom':
        # 方法4：自定义权重
        cls_weights = np.array([1, 80, 3], np.float32)

    else:
        raise ValueError(f"Unknown method: {method}")

    print("\nComputed Class Weights:")
    for c in range(len(cls_weights)):
        print(f"Class {c}: {cls_weights[c]:.4f}")
    # cls_weights[2] = 1.1
    # cls_weights[1] = 200

    return cls_weights


# 修改LossHistory类以记录准确率
class EnhancedLossHistory(LossHistory):
    def __init__(self, log_dir, model, input_shape):
        super(EnhancedLossHistory, self).__init__(log_dir, model, input_shape)
        self.acc_details = {
            "train_seg_acc": [],
            "val_seg_acc": [],
            "domain_acc": [],
            "domain_acc_backbone_source": [],
            "domain_acc_backbone_target": [],
            "domain_acc_aspp_source": [],
            "domain_acc_aspp_target": []
        }

    def append_acc(self, epoch, train_seg_acc, val_seg_acc, domain_acc,
                   domain_acc_backbone_source, domain_acc_backbone_target,
                   domain_acc_aspp_source, domain_acc_aspp_target):
        self.acc_details["train_seg_acc"].append(train_seg_acc)
        self.acc_details["val_seg_acc"].append(val_seg_acc)
        self.acc_details["domain_acc"].append(domain_acc)
        self.acc_details["domain_acc_backbone_source"].append(domain_acc_backbone_source)
        self.acc_details["domain_acc_backbone_target"].append(domain_acc_backbone_target)
        self.acc_details["domain_acc_aspp_source"].append(domain_acc_aspp_source)
        self.acc_details["domain_acc_aspp_target"].append(domain_acc_aspp_target)

        with open(os.path.join(self.log_dir, "acc_epoch.txt"), 'a') as f:
            f.write(str(epoch))
            f.write("," + str(train_seg_acc))
            f.write("," + str(val_seg_acc))
            f.write("," + str(domain_acc))
            f.write("," + str(domain_acc_backbone_source))
            f.write("," + str(domain_acc_backbone_target))
            f.write("," + str(domain_acc_aspp_source))
            f.write("," + str(domain_acc_aspp_target))
            f.write("\n")

        self.acc_plot()

    def acc_plot(self):
        iters = range(len(self.acc_details["train_seg_acc"]))

        plt.figure()
        plt.plot(iters, self.acc_details["train_seg_acc"], 'red', linewidth=2, label='train seg acc')
        plt.plot(iters, self.acc_details["val_seg_acc"], 'coral', linewidth=2, label='val seg acc')

        try:
            if len(self.acc_details["train_seg_acc"]) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.acc_details["train_seg_acc"], num, 3), 'green',
                     linestyle='--', linewidth=2, label='smooth train seg acc')
            plt.plot(iters, scipy.signal.savgol_filter(self.acc_details["val_seg_acc"], num, 3), '#8B4513',
                     linestyle='--', linewidth=2, label='smooth val seg acc')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.acc_details["domain_acc"], 'blue', linewidth=2, label='domain acc')
        plt.plot(iters, self.acc_details["domain_acc_backbone_source"], 'green', linewidth=2,
                 label='domain acc backbone source')
        plt.plot(iters, self.acc_details["domain_acc_backbone_target"], 'red', linewidth=2,
                 label='domain acc backbone target')
        plt.plot(iters, self.acc_details["domain_acc_aspp_source"], 'cyan', linewidth=2, label='domain acc aspp source')
        plt.plot(iters, self.acc_details["domain_acc_aspp_target"], 'magenta', linewidth=2,
                 label='domain acc aspp target')

        try:
            if len(self.acc_details["domain_acc"]) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.acc_details["domain_acc"], num, 3), 'darkblue',
                     linestyle='--', linewidth=2, label='smooth domain acc')
            plt.plot(iters, scipy.signal.savgol_filter(self.acc_details["domain_acc_backbone_source"], num, 3),
                     'lightgreen',
                     linestyle='--', linewidth=2, label='smooth domain acc backbone source')
            plt.plot(iters, scipy.signal.savgol_filter(self.acc_details["domain_acc_backbone_target"], num, 3),
                     'lightcoral',
                     linestyle='--', linewidth=2, label='smooth domain acc backbone target')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Domain Acc')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_domain_acc.png"))

        plt.cla()
        plt.close("all")


# 主函数
if __name__ == "__main__":
    # ---------------------------------#
    #   基本配置参数
    # ---------------------------------#
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = False
    num_classes = 2
    backbone = "ghostnet"
    pretrained = False
    model_path = ""  # 完整的DeepLabV3+预训练模型
    downsample_factor = 16
    input_shape = [640, 640]

    # ---------------------------------#
    #   半监督DANN特定参数
    # ---------------------------------#
    use_dann = True  # 是否使用DANN
    lambda_domain = 0.5  # 域对抗损失权重
    lambda_supervised = 1.0  # 有标注目标域分割损失权重

    # 数据集路径配置
    source_VOCdevkit_path = 'VOCdevkit_1-2仁'  # 源域数据集路径
    target_VOCdevkit_path = 'target_voc'  # 目标域数据集路径

    # 目标域有标注和无标注数据分割比例
    target_labeled_ratio = 0.034  # 30%的目标域数据有标注

    # ---------------------------------#
    #   训练参数
    # ---------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 1
    UnFreeze_Epoch = 500  # 增加训练周期
    Unfreeze_batch_size = 1
    Freeze_Train = False
    Init_lr = 5e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 1e-4
    lr_decay_type = 'cos'
    save_period = 1
    save_dir = 'logs'
    eval_flag = True
    eval_period = 10
    dice_loss = False
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)

    num_workers = 4

    # 设置随机种子
    seed_everything(seed)

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # ----------------------------------------------------#
    #   下载预训练权重
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    # ----------------------------------------------------#
    #   创建模型（半监督DANN版本）
    # ----------------------------------------------------#
    if True:
        model = DeepLabDANN(num_classes=num_classes, backbone=backbone,
                            downsample_factor=downsample_factor, pretrained=pretrained,
                            lambda_domain=lambda_domain)
        if local_rank == 0:
            print("Using Semi-Supervised DANN model for domain adaptation")
    else:
        model = DeepLab(num_classes=num_classes, backbone=backbone,
                        downsample_factor=downsample_factor, pretrained=pretrained)
        if local_rank == 0:
            print("Using standard DeepLabV3+ model")

    # ----------------------------------------------------#
    #   加载预训练权重
    # ----------------------------------------------------#
    if not pretrained:
        weights_init(model)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)

        # 对于DANN模型，我们需要加载完整的DeepLabV3+权重
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            # 尝试直接匹配键
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                # 尝试匹配DANN模型中的DeepLab部分
                if k.startswith('deeplab.'):
                    new_k = k[8:]  # 去掉'deeplab.'前缀
                    if new_k in model_dict.keys() and np.shape(model_dict[new_k]) == np.shape(v):
                        temp_dict[new_k] = v
                        load_key.append(new_k)
                else:
                    no_load_key.append(k)

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            if no_load_key:
                print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # ----------------------#
    #   记录Loss
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        # 使用增强的LossHistory类
        loss_history = EnhancedLossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    # 源域数据集
    with open(os.path.join(source_VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        source_train_lines = f.readlines()
    with open(os.path.join(source_VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        source_val_lines = f.readlines()

    # 目标域数据集 - 分割为有标注和无标注两部分
    with open(os.path.join(target_VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        target_all_lines = f.readlines()

    # 随机打乱目标域数据
    np.random.seed(seed)
    np.random.shuffle(target_all_lines)

    # 计算有标注数据量
    num_target_labeled = int(len(target_all_lines) * target_labeled_ratio)
    target_labeled_lines = target_all_lines[:num_target_labeled]
    target_unlabeled_lines = target_all_lines[num_target_labeled:]

    if local_rank == 0:
        print(f"目标域数据分割: 有标注 {len(target_labeled_lines)} 张, 无标注 {len(target_unlabeled_lines)} 张")

    num_train = len(source_train_lines)
    num_val = len(source_val_lines)

    # 计算类别权重（使用源域数据）
    if local_rank == 0:
        print("\nCalculating class weights for source domain...")
        cls_weights = np.ones([num_classes], np.float32)
    else:
        # 使用默认权重（将在后续广播）
        cls_weights = np.ones([num_classes], np.float32)

    # 广播类别权重到所有进程（分布式训练）
    if distributed:
        cls_weights = torch.from_numpy(cls_weights).float()
        dist.broadcast(cls_weights, src=0)
        cls_weights = cls_weights.numpy()

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        # 打印半监督特定配置
        print(f"Semi-Supervised DANN Configuration:")
        print(f"  - Target Labeled Ratio: {target_labeled_ratio}")
        print(f"  - Labeled Target Samples: {len(target_labeled_lines)}")
        print(f"  - Unlabeled Target Samples: {len(target_unlabeled_lines)}")
        print(f"  - Lambda Domain: {lambda_domain}")
        print(f"  - Lambda Supervised: {lambda_supervised}")

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # -------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        # 对于DANN模型，我们需要优化所有参数（包括域分类器）
        if use_dann:
            params_to_optimize = model_train.parameters()
        else:
            params_to_optimize = model_train.parameters()

        optimizer = {
            'adam': optim.AdamW(params_to_optimize, Init_lr_fit, betas=(momentum, 0.9), weight_decay=weight_decay),
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # 创建数据集
        source_train_dataset = DeeplabDataset(source_train_lines, input_shape, num_classes, True, source_VOCdevkit_path)
        target_labeled_dataset = DeeplabDataset(target_labeled_lines, input_shape, num_classes, True,
                                                target_VOCdevkit_path)
        target_unlabeled_dataset = DeeplabDataset(target_unlabeled_lines, input_shape, num_classes, True,
                                                  target_VOCdevkit_path, True)  # 无标注数据
        val_dataset = DeeplabDataset(source_val_lines, input_shape, num_classes, False, source_VOCdevkit_path)

        if distributed:
            source_train_sampler = torch.utils.data.distributed.DistributedSampler(source_train_dataset, shuffle=True)
            target_labeled_sampler = torch.utils.data.distributed.DistributedSampler(target_labeled_dataset,
                                                                                     shuffle=True)
            target_unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(target_unlabeled_dataset,
                                                                                       shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            source_train_sampler = None
            target_labeled_sampler = None
            target_unlabeled_sampler = None
            val_sampler = None
            shuffle = True

        # 创建数据加载器
        gen_source = DataLoader(source_train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                                sampler=source_train_sampler,
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        gen_target_labeled = DataLoader(target_labeled_dataset, shuffle=shuffle, batch_size=batch_size,
                                        num_workers=num_workers, pin_memory=True, drop_last=True,
                                        collate_fn=deeplab_dataset_collate, sampler=target_labeled_sampler,
                                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        gen_target_unlabeled = DataLoader(target_unlabeled_dataset, shuffle=shuffle, batch_size=batch_size,
                                          num_workers=num_workers, pin_memory=True, drop_last=True,
                                          collate_fn=deeplab_dataset_collate, sampler=target_unlabeled_sampler,
                                          worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                             sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(model_train, input_shape, num_classes, source_val_lines, source_VOCdevkit_path,
                                         log_dir, Cuda, eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 训练一个epoch
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch_semi_supervised_dann(model_train, model, loss_history, eval_callback, optimizer, epoch,
                                               epoch_step, epoch_step_val, gen_source, gen_target_labeled,
                                               gen_target_unlabeled, gen_val,
                                               UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes,
                                               fp16, scaler, save_period, save_dir, local_rank,
                                               lambda_domain, lambda_supervised)

        if local_rank == 0:
            loss_history.writer.close()