import datetime
import os
from functools import partial

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


# 新增：DANN版本的DeepLabV3+模型
class DeepLabDANN(torch.nn.Module):
    def __init__(self, num_classes, backbone, downsample_factor, pretrained, lambda_domain=0.5):
        super(DeepLabDANN, self).__init__()
        # 原始DeepLabV3+模型
        self.deeplab = DeepLab(num_classes=num_classes, backbone=backbone,
                               downsample_factor=downsample_factor, pretrained=pretrained)

        # 域分类器 - 使用ASPP输出的通道数(256)，而不是backbone的输出通道数
        domain_input_channels = 128  # ASPP输出特征通道数固定为256

        self.domain_classifier = DomainClassifier(domain_input_channels)
        self.lambda_domain = lambda_domain  # 域对抗损失权重
        self.alpha = 0  # GRL参数，将在训练中动态调整

    def forward(self, x, alpha=None, mode='train'):
        if alpha is None:
            alpha = self.alpha

        # 获取输入图像尺寸
        H, W = x.size(2), x.size(3)

        # 获取DeepLabV3+的特征
        low_level_features, x_backbone = self.deeplab.backbone(x)
        # x_lrsa = self.deeplab.aspp_lrsa(x_backbone)
        x_aspp = self.deeplab.aspp(x_backbone)

        # 继续分割解码过程
        low_level_features = self.deeplab.shortcut_conv(low_level_features)
        x_aspp = F.interpolate(x_aspp, size=(low_level_features.size(2), low_level_features.size(3)),
                               mode='bilinear', align_corners=True)
        cls_conv_before = self.deeplab.cat_conv(torch.cat((x_aspp, low_level_features), dim=1))
        x = self.deeplab.cls_conv(cls_conv_before)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        # 如果是训练模式，计算域分类损失
        if mode == 'train':
            # 应用梯度反转层到ASPP特征
            reversed_features = GradientReversalLayer.apply(x_aspp, alpha)
            # reversed_features = GradientReversalLayer.apply(cls_conv_before, alpha)
            domain_output = self.domain_classifier(reversed_features)
            return x, domain_output

        else:
            # 验证/测试模式：直接返回分割结果
            return x

    def set_alpha(self, alpha):
        self.alpha = alpha


# 修改后的训练函数（DANN版本）
def fit_one_epoch_dann(model_train, model, loss_history, eval_callback, optimizer, epoch,
                       epoch_step, epoch_step_val, gen_source, gen_target, gen_val, UnFreeze_Epoch, Cuda,
                       dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                       save_period, save_dir, local_rank, lambda_domain=0.5):
    total_loss = 0
    total_seg_loss = 0
    total_domain_loss = 0
    val_loss = 0

    # 新增：记录域分类准确率
    total_domain_acc_source = 0
    total_domain_acc_target = 0
    total_domain_acc = 0

    # 新增：记录分割准确率
    total_seg_acc = 0
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
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}', postfix=dict, mininterval=0.3)

    # 设置模型为训练模式
    model_train.train()

    # 创建源域和目标域数据迭代器
    source_iter = iter(gen_source)
    target_iter = iter(gen_target)

    for iteration in range(epoch_step):
        try:
            # 获取源域数据
            source_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(gen_source)
            source_batch = next(source_iter)

        try:
            # 获取目标域数据
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(gen_target)
            target_batch = next(target_iter)

        # 处理源域数据
        imgs_source, pngs_source, labels_source = source_batch
        domain_labels_source = torch.zeros(imgs_source.size(0), dtype=torch.long)  # 源域标签为0

        # 处理目标域数据
        imgs_target, pngs_target, labels_target = target_batch
        domain_labels_target = torch.ones(imgs_target.size(0), dtype=torch.long)  # 目标域标签为1

        if Cuda:
            imgs_source = imgs_source.cuda()
            pngs_source = pngs_source.cuda()
            domain_labels_source = domain_labels_source.cuda()

            imgs_target = imgs_target.cuda()
            domain_labels_target = domain_labels_target.cuda()

        with torch.cuda.amp.autocast(enabled=fp16):
            # 源域前向传播
            # 修改后：

            # print('源域前向传播...')
            result = model_train(imgs_source, alpha=alpha, mode='train')
            seg_output_source, domain_output_source = result

            # 目标域前向传播
            # print('目标域前向传播...')
            _, domain_output_target = model_train(imgs_target, alpha=alpha, mode='train')

            # print('计算分割损失（仅源域）...')
            # 计算分割损失（仅源域）
            seg_loss = compute_segmentation_loss(seg_output_source, pngs_source, weights=cls_weights,
                                                 num_classes=num_classes, dice_loss=dice_loss,
                                                 focal_loss=focal_loss)
            # print('计算域分类损失（源域+目标域）...')
            # 计算域分类损失（源域+目标域）
            domain_loss_source = torch.nn.functional.cross_entropy(domain_output_source, domain_labels_source)
            domain_loss_target = torch.nn.functional.cross_entropy(domain_output_target, domain_labels_target)

            # print(f'domain_loss_source={domain_loss_source.size()},domain_loss_target_len={domain_loss_target.size()}')
            domain_loss = domain_loss_source + domain_loss_target

            # 总损失 = 分割损失 + λ * 域对抗损失
            loss = seg_loss + lambda_domain * domain_loss
            # 新增：计算域分类准确率
            domain_pred_source = torch.argmax(domain_output_source, dim=1)
            domain_acc_source = (domain_pred_source == domain_labels_source).float().mean()

            domain_pred_target = torch.argmax(domain_output_target, dim=1)
            domain_acc_target = (domain_pred_target == domain_labels_target).float().mean()
            domain_acc = (domain_acc_source + domain_acc_target) / 2

            # 新增：计算分割准确率
            seg_pred = torch.argmax(seg_output_source, dim=1)
            seg_acc = (seg_pred == pngs_source).float()
            # 忽略标签为255的像素（如果有）
            mask = pngs_source != 255
            seg_acc = seg_acc[mask].mean() if mask.any() else torch.tensor(0.0).to(seg_acc.device)

            # 新增：计算每个类别的准确率
            for c in range(num_classes):
                mask_c = (pngs_source == c)
                if mask_c.any():
                    correct_c = (seg_pred[mask_c] == pngs_source[mask_c]).float().sum().item()
                    class_correct[c] += correct_c
                    class_total[c] += mask_c.sum().item()

                    # 计算每个类别的IoU
                    pred_c = (seg_pred == c)
                    target_c = (pngs_source == c)

                    intersection = (pred_c & target_c).float().sum().item()
                    union = (pred_c | target_c).float().sum().item()

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
        total_domain_loss += domain_loss.item()

        # 新增：累加准确率
        total_domain_acc_source += domain_acc_source.item()
        total_domain_acc_target += domain_acc_target.item()
        total_domain_acc += domain_acc.item()
        total_seg_acc += seg_acc.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'seg_loss': total_seg_loss / (iteration + 1),
                                'domain_loss': total_domain_loss / (iteration + 1),
                                'domain_acc': total_domain_acc / (iteration + 1),
                                'seg_acc': total_seg_acc / (iteration + 1),
                                'alpha': alpha,
                                'lr': get_lr(optimizer)})
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
                                total_domain_acc_source / epoch_step,
                                total_domain_acc_target / epoch_step)

        print('Epoch:' + str(epoch + 1) + '/' + str(UnFreeze_Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        print('Domain Acc: %.3f%% (Source: %.3f%%, Target: %.3f%%)' %
              (total_domain_acc / epoch_step * 100,
               total_domain_acc_source / epoch_step * 100,
               total_domain_acc_target / epoch_step * 100))

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
            torch.save(model.state_dict(), os.path.join(save_dir, 'pth/ep%03d-loss%.3f-val_loss%.3f.pth' %
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
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
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
            "domain_acc_source": [],
            "domain_acc_target": []
        }

    def append_acc(self, epoch, train_seg_acc, val_seg_acc, domain_acc, domain_acc_source, domain_acc_target):
        self.acc_details["train_seg_acc"].append(train_seg_acc)
        self.acc_details["val_seg_acc"].append(val_seg_acc)
        self.acc_details["domain_acc"].append(domain_acc)
        self.acc_details["domain_acc_source"].append(domain_acc_source)
        self.acc_details["domain_acc_target"].append(domain_acc_target)

        with open(os.path.join(self.log_dir, "acc_epoch.txt"), 'a') as f:
            f.write(str(epoch))
            f.write("," + str(train_seg_acc))
            f.write("," + str(val_seg_acc))
            f.write("," + str(domain_acc))
            f.write("," + str(domain_acc_source))
            f.write("," + str(domain_acc_target))
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
        plt.plot(iters, self.acc_details["domain_acc_source"], 'green', linewidth=2, label='domain acc source')
        plt.plot(iters, self.acc_details["domain_acc_target"], 'red', linewidth=2, label='domain acc target')

        try:
            if len(self.acc_details["domain_acc"]) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.acc_details["domain_acc"], num, 3), 'cyan',
                     linestyle='--',
                     linewidth=2, label='smooth domain acc')
            plt.plot(iters, scipy.signal.savgol_filter(self.acc_details["domain_acc_source"], num, 3), '#FFA500',
                     linestyle='--', linewidth=2, label='smooth domain acc source')
            plt.plot(iters, scipy.signal.savgol_filter(self.acc_details["domain_acc_target"], num, 3), '#800080',
                     linestyle='--', linewidth=2, label='smooth domain acc target')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Domain Acc')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_domain_acc.png"))

        plt.cla()
        plt.close("all")


# 可视化工具类
class VisualizationTool:
    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape

        # 设置颜色映射
        if num_classes == 3:
            self.colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]  # 背景、类别1、类别2
        else:
            # 生成随机颜色
            hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                                   hsv_tuples))
            self.colors[0] = (0, 0, 0)  # 背景设为黑色

    def detect_image(self, model, image):
        """
        对单张图像进行预测并返回分割结果
        """
        # 调整图像尺寸
        iw, ih = image.size
        image_data, (pad_w, pad_h, scale, (new_w, new_h)) = self.resize_image_with_info(
            image, (self.input_shape[1], self.input_shape[0])
        )

        # 添加批次维度并转换为tensor
        image_data_np = np.array(image_data, np.float32) / 255.0
        image_data_np = np.transpose(image_data_np, (2, 0, 1))
        image_data = np.expand_dims(image_data_np, 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if torch.cuda.is_available():
                images = images.cuda()

            # 模型预测
            outputs = model(images, mode='val')
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            pr = outputs.cpu().numpy()

            # 获取预测结果
            pr = pr[0]
            pr = pr.transpose(1, 2, 0)
            pr = np.argmax(pr, axis=-1)

            # 关键修改：正确处理填充区域的掩码
            if pad_h > 0 or pad_w > 0:
                # 计算有效区域（去除填充）
                valid_h = self.input_shape[0] - 2 * pad_h
                valid_w = self.input_shape[1] - 2 * pad_w
                pr = pr[pad_h:pad_h + valid_h, pad_w:pad_w + valid_w]

            # 将有效区域的预测缩放到原始图像尺寸
            pr = cv2.resize(pr.astype(np.float32), (iw, ih),
                            interpolation=cv2.INTER_NEAREST)
            pr = pr.astype(np.int64)

        # 生成分割图像
        seg_img = np.zeros((pr.shape[0], pr.shape[1], 3), dtype=np.uint8)
        for c in range(self.num_classes):
            mask = (pr == c)
            seg_img[mask] = self.colors[c]

        return Image.fromarray(seg_img)

    def resize_image_with_info(self, image, size):
        """
        改进的resize方法，返回图像和变换信息
        返回: (resized_image, (pad_left, pad_top, scale, (new_width, new_height)))
        """
        iw, ih = image.size
        w, h = size

        # 计算缩放比例（保持长宽比）
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        # 计算填充
        pad_w = (w - nw) // 2
        pad_h = (h - nh) // 2

        # 使用与训练时相同的插值方法
        image_resized = image.resize((nw, nh), Image.BILINEAR)

        # 使用与训练时相同的填充颜色
        fill_color = (0, 0, 0)  # 黑色填充
        new_image = Image.new('RGB', (w, h), fill_color)
        new_image.paste(image_resized, (pad_w, pad_h))

        return new_image, (pad_w, pad_h, scale, (nw, nh))

    def visualize_results(self, model, lines, VOCdevkit_path, save_dir, num_visual=5, domain='source'):
        """
        可视化结果并保存 - 批处理版本
        domain: 'source'或'target'，指定是源领域还是目标领域
        """
        os.makedirs(save_dir, exist_ok=True)
        print(f"批量保存 {domain} 领域的可视化结果到 {save_dir}，数量: {min(num_visual, len(lines))}")

        # 随机选择num_visual个样本进行可视化
        indices = np.random.choice(len(lines), min(num_visual, len(lines)), replace=False)

        # 批量处理：一次性读取所有图像
        images = []
        names = []
        ground_truths = []  # 用于源领域的真实标签

        print("批量读取图像数据...")
        for i in indices:
            line = lines[i]
            name = line.split()[0]
            names.append(name)

            # 加载图像
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages", name + ".jpg")
            image = Image.open(image_path)
            image = image.convert('RGB')
            images.append(image)

            # 如果是源领域，加载真实标签
            if domain == 'source':
                label_path = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass", name + ".png")
                label = Image.open(label_path)
                ground_truths.append(label)

        # 批量预测分割结果
        print("批量进行分割预测...")
        seg_imgs = []
        for image in images:
            seg_img = self.detect_image(model, image)
            seg_imgs.append(seg_img)

        # 批量创建和保存对比图像
        print("批量生成可视化结果...")
        for i, (name, image, seg_img) in enumerate(zip(names, images, seg_imgs)):
            if domain == 'source':
                # 源领域：显示原始图像、真实标签和预测结果
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # 原始图像
                axes[0].imshow(np.array(image))
                axes[0].set_title(f'Original Image: {name}')
                axes[0].axis('off')

                # 真实标签
                label_array = np.array(ground_truths[i])
                label_vis = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
                for c in range(self.num_classes):
                    mask = (label_array == c)
                    label_vis[mask] = self.colors[c]
                axes[1].imshow(label_vis)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                # 预测结果
                axes[2].imshow(np.array(seg_img))
                axes[2].set_title('Prediction')
                axes[2].axis('off')
            else:
                # 目标领域：只显示原始图像和预测结果
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                # 原始图像
                axes[0].imshow(np.array(image))
                axes[0].set_title(f'Original Image: {name}')
                axes[0].axis('off')

                # 预测结果
                axes[1].imshow(np.array(seg_img))
                axes[1].set_title('Prediction')
                axes[1].axis('off')

            # 保存对比图像
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"{name}_comparison.jpg"),
                dpi=200,
                bbox_inches='tight',
                facecolor='white'  # 确保背景为白色[8](@ref)
            )
            plt.close()  # 关闭图形释放内存[6,7](@ref)

            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == len(indices):
                print(f"已处理 {i + 1}/{len(indices)} 张图像")

        print(f"批量可视化完成！所有结果已保存到: {save_dir}")


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
    model_path = "logs/cheaplab_dann_7_batchsize_4_640_no_star/ep010-loss0.708-val_loss0.011.pth"  # 完整的DeepLabV3+预训练模型
    downsample_factor = 16
    input_shape = [640, 640]

    # ---------------------------------#
    #   DANN特定参数
    # ---------------------------------#
    use_dann = True  # 是否使用DANN
    lambda_domain = 0.5  # 域对抗损失权重

    # 数据集路径配置
    source_VOCdevkit_path = 'F:\BaiduNetdiskDownload\VOCdevkit_1-2仁'  # 源域数据集路径
    # source_VOCdevkit_path = 'F:\BaiduNetdiskDownload\\1-2仁'  # 源域数据集路径
    target_VOCdevkit_path = 'F:/BaiduNetdiskDownload/板栗/archive/chestnut_zonguldak'  # 目标域数据集路径
    # target_VOCdevkit_path = 'F:\BaiduNetdiskDownload\板栗\\archive\chestnut_improve'  # 目标域数据集路径

    # ---------------------------------#
    #   训练参数
    # ---------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 4
    UnFreeze_Epoch = 500  # 增加训练周期
    Unfreeze_batch_size = 4
    Freeze_Train = False
    Init_lr = 5e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 1e-4
    lr_decay_type = 'cos'
    save_period = 1
    save_dir = 'logs/cheaplab_dann_8_640_vistarget_600'
    eval_flag = True
    eval_period = 10
    dice_loss = False
    focal_loss = False  # 启用focal_loss
    # cls_weights = np.ones([num_classes], np.float32)
    # cls_weights = np.array([1, 2, 2], np.float32)

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
    #   创建模型（DANN或原始版本）
    # ----------------------------------------------------#
    if True:
        model = DeepLabDANN(num_classes=num_classes, backbone=backbone,
                            downsample_factor=downsample_factor, pretrained=pretrained,
                            lambda_domain=lambda_domain)
        if local_rank == 0:
            print("Using DANN model for domain adaptation")
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

    # 目标域数据集
    with open(os.path.join(target_VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        target_train_lines = f.readlines()

    num_train = len(source_train_lines)
    num_val = len(source_val_lines)

    # 计算类别权重（只使用源域数据）
    if local_rank == 0:
        print("\nCalculating class weights for source domain...")
        source_class_ratios = calculate_class_weights(source_train_lines, source_VOCdevkit_path, num_classes)
        #
        # 使用源域的类别分布计算权重
        # 使用倒数方法计算权重
        cls_weights = compute_class_weights(source_class_ratios, method='median_frequency')
        #
        print("\nFinal Class Weights:")
        for c in range(num_classes):
            print(f"Class {c}: {cls_weights[c]:.4f}")
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
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        # ----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (
                optimizer_type, wanted_step))
            print(
                "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                    num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
                total_step, wanted_step, wanted_epoch))

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

        # optimizer = {
        #     'adam': optim.Adam(params_to_optimize, Init_lr_fit, betas=(momentum, 0.9), weight_decay=weight_decay),
        # }[optimizer_type]
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
        target_train_dataset = DeeplabDataset(target_train_lines, input_shape, num_classes, True, target_VOCdevkit_path,
                                              True)
        val_dataset = DeeplabDataset(source_val_lines, input_shape, num_classes, False, source_VOCdevkit_path)

        if distributed:
            source_train_sampler = torch.utils.data.distributed.DistributedSampler(source_train_dataset, shuffle=True)
            target_train_sampler = torch.utils.data.distributed.DistributedSampler(target_train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            source_train_sampler = None
            target_train_sampler = None
            val_sampler = None
            shuffle = True

        # 创建数据加载器
        gen_source = DataLoader(source_train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                                sampler=source_train_sampler,
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        gen_target = DataLoader(target_train_dataset, shuffle=shuffle, batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                                sampler=target_train_sampler,
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                             sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        if local_rank == 0:
            # 使用新的EvalCallback类
            eval_callback = EvalCallback(model, input_shape, num_classes, source_val_lines,
                                         source_VOCdevkit_path,
                                         log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # 创建可视化工具
        visual_tool = VisualizationTool(num_classes, input_shape)

        # 创建训练集可视化目录
        train_visual_dir = os.path.join(save_dir, "train_visual")
        os.makedirs(train_visual_dir, exist_ok=True)

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

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
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen_source = DataLoader(source_train_dataset, shuffle=shuffle, batch_size=batch_size,
                                        num_workers=num_workers,
                                        pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                                        sampler=source_train_sampler,
                                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                gen_target = DataLoader(target_train_dataset, shuffle=shuffle, batch_size=batch_size,
                                        num_workers=num_workers,
                                        pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                                        sampler=target_train_sampler,
                                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                                     sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                if source_train_sampler is not None:
                    source_train_sampler.set_epoch(epoch)
                if target_train_sampler is not None:
                    target_train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # 使用DANN训练函数或原始训练函数
            if use_dann:
                fit_one_epoch_dann(model_train, model, loss_history, eval_callback, optimizer, epoch,
                                   epoch_step, epoch_step_val, gen_source, gen_target, gen_val, UnFreeze_Epoch, Cuda,
                                   dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                                   save_period, save_dir, local_rank, lambda_domain=lambda_domain)
            else:
                # 这里需要您提供原始的训练函数fit_one_epoch
                # 由于原始代码中可能已经定义，这里暂时用pass代替
                print("原始训练函数需要您根据原始代码提供")
                break

            # 在每个epoch结束后保存训练集的可视化结果
            if local_rank == 0:
                epoch_train_visual_dir = os.path.join(train_visual_dir, f"epoch_{epoch + 1}")
                os.makedirs(epoch_train_visual_dir, exist_ok=True)

                # 创建源领域和目标领域的子目录
                source_visual_dir = os.path.join(epoch_train_visual_dir, "source")
                target_visual_dir = os.path.join(epoch_train_visual_dir, "target")
                os.makedirs(source_visual_dir, exist_ok=True)
                os.makedirs(target_visual_dir, exist_ok=True)

                # 可视化源领域结果
                visual_tool.visualize_results(
                    model_train,
                    source_train_lines,
                    source_VOCdevkit_path,
                    source_visual_dir,
                    20,  # 每个epoch可视化30个训练样本
                    domain='source'
                )

                # 可视化目标领域结果
                visual_tool.visualize_results(
                    model_train,
                    target_train_lines,
                    target_VOCdevkit_path,
                    target_visual_dir,
                    600,  # 每个epoch可视化30个训练样本
                    domain='target'
                )

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
