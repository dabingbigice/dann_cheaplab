import datetime
import os
import random
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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from nets.deeplabv3_plus import DeepLab

# 设置matplotlib后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal


# ==============================
# 1. 自定义DeeplabDataset（添加文件名返回）
# ==============================
class DeeplabDataset(Dataset):
    def __init__(self, lines, input_shape, num_classes, train, dataset_path, is_target=False):
        super(DeeplabDataset, self).__init__()
        self.lines = lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.is_target = is_target  # 标记是否为目标域数据集

        # 数据预处理（保持你的原有逻辑）
        self.transform = self._get_transform()

    def _get_transform(self):
        """获取数据预处理管道（保持你的原有逻辑）"""
        import torchvision.transforms as transforms
        transform_list = [
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ]
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index].strip()
        if not line:
            raise ValueError(f"Dataset lines[{index}] is empty! 检查txt文件是否有空白行")
        file_name = line.split()[0]

        # 加载图像
        img_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", f"{file_name}.jpg")
        img = Image.open(img_path).convert("RGB")

        # 加载掩码
        if not self.is_target:
            mask_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", f"{file_name}.png")
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", img.size, 0)

        # 预处理：图像和 mask 都缩放为 320x320（关键修复）
        img = self.transform(img)
        # 给 mask 单独做相同的 Resize 变换（保持和图像尺寸一致）
        mask_transform = transforms.Compose([transforms.Resize((320, 320), interpolation=Image.NEAREST)])
        mask = mask_transform(mask)
        mask = transforms.ToTensor()(mask).long().squeeze(0)  # 转为 (320,320) 格式

        label = torch.tensor(0, dtype=torch.long)
        return img, mask, label, file_name


# ==============================
# 2. 自定义collate_fn（拼接文件名）
# ==============================
def deeplab_dataset_collate(batch):
    """
    拼接batch，包含文件名列表
    input: batch = [(img1, mask1, label1, file1), (img2, mask2, label2, file2), ...]
    output: (imgs_batch, masks_batch, labels_batch, file_names_batch)
    """
    imgs = []
    pngs = []
    labels = []
    file_names = []  # 新增：收集文件名

    for img, png, label, file_name in batch:
        imgs.append(img)
        pngs.append(png)
        labels.append(label)
        file_names.append(file_name)  # 拼接每个样本的文件名

    # 转换为批次张量
    imgs = torch.stack(imgs)
    pngs = torch.stack(pngs)
    labels = torch.stack(labels)

    # 关键：返回文件名列表
    return imgs, pngs, labels, file_names


# ==============================
# 3. 原有辅助类/函数（保持不变，补充缺失依赖）
# ==============================


class LossHistory:
    """占位：原始LossHistory类"""

    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.writer = None

    def append_loss(self, epoch, train_loss, val_loss):
        pass


class EvalCallback:
    """占位：原始EvalCallback类"""

    def __init__(self, model, input_shape, num_classes, val_lines, dataset_path, log_dir, cuda, eval_flag=True,
                 period=10):
        pass

    def on_epoch_end(self, epoch, model):
        pass


def get_lr_scheduler(lr_decay_type, init_lr, min_lr, epochs):
    """占位：学习率调度器"""

    def scheduler(epoch):
        return init_lr * (min_lr / init_lr) ** (epoch / epochs)

    return scheduler


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """设置学习率"""
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weights_init(model):
    """权重初始化"""
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


def download_weights(backbone):
    """下载预训练权重（占位）"""
    pass


def seed_everything(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def show_config(**kwargs):
    """打印配置信息"""
    for k, v in kwargs.items():
        print(f"{k}: {v}")


def worker_init_fn(worker_id, rank, seed):
    """worker初始化函数"""
    np.random.seed(seed + worker_id + rank)


# ==============================
# 4. 原有核心功能（DANN相关，保持不变，修正文件名获取逻辑）
# ==============================
ration = 3
# 目标域文件名列表（无需后缀）
TARGET_FILENAMES = {
    "photo_20251115_114144_590864",
    "photo_20251115_114150_354268",
    "photo_20251115_114156_099701",
    "photo_20251115_114204_099638",
    "photo_20251115_114239_459843",
    "photo_20251115_114251_430968",
    "photo_20251115_114256_788783",
    "photo_20251115_114301_648240",
    "photo_20251115_114347_590810",
    "photo_20251115_114354_885429",
    "photo_20251115_114359_590002",
    "photo_20251115_114406_052534",
    "photo_20251115_114427_492863",
    "photo_20251115_114432_303377",
    "photo_20251115_114437_956730",
    "photo_20251115_114442_520689",
    "photo_20251115_114542_916763",
    "photo_20251115_114546_998101",
    "photo_20251115_114555_221134",
    "photo_20251115_114559_430473",
    "photo_20251115_114613_230848",
    "photo_20251115_114617_990749",
    "photo_20251115_114621_781525",
    "photo_20251115_114625_684686",
    "photo_20251115_114645_060958",
}


# 梯度反转层（GRL）实现
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# DANN域分类器
class DomainClassifier(torch.nn.Module):
    def __init__(self, input_channels=128, hidden_size=1024, num_domains=2):
        super(DomainClassifier, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_channels, num_domains),
        )
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
        # 确保输入和模型在同一设备上
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# DANN版本的DeepLabV3+模型
class DeepLabDANN(torch.nn.Module):
    def __init__(self, num_classes, backbone, downsample_factor, pretrained, lambda_domain=0.5):
        super(DeepLabDANN, self).__init__()
        self.deeplab = DeepLab(num_classes=num_classes, backbone=backbone,
                               downsample_factor=downsample_factor, pretrained=pretrained)

        # 获取backbone的输出通道数
        if backbone == 'ghostnet':
            backbone_output_channels = 96

        # 域分类器
        self.domain_classifier_backbone = DomainClassifier(backbone_output_channels)
        aspp_output_channels = 128
        self.domain_classifier_aspp = DomainClassifier(12)

        self.lambda_domain = lambda_domain
        self.alpha = 0

    def forward(self, x, alpha=None, mode='train'):
        if alpha is None:
            alpha = self.alpha

        H, W = x.size(2), x.size(3)
        low_level_features_backbone, x_backbone = self.deeplab.backbone(x)

        x_aspp = self.deeplab.aspp(x_backbone)
        low_level_features = self.deeplab.shortcut_conv(low_level_features_backbone)
        x_aspp = torch.nn.functional.interpolate(x_aspp, size=(low_level_features.size(2), low_level_features.size(3)),
                                                 mode='bilinear', align_corners=True)
        cls_conv_before = self.deeplab.cat_conv(torch.cat((x_aspp, low_level_features), dim=1))
        x = self.deeplab.cls_conv(cls_conv_before)
        x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        if mode == 'train':
            reversed_features_backbone = GradientReversalLayer.apply(x_backbone, alpha)
            domain_output_backbone = self.domain_classifier_backbone(reversed_features_backbone)

            reversed_features_aspp = GradientReversalLayer.apply(x_aspp, alpha)
            domain_output_aspp = self.domain_classifier_aspp(low_level_features_backbone)

            return x, domain_output_backbone, domain_output_aspp
        else:
            return x

    def set_alpha(self, alpha):
        self.alpha = alpha


# 辅助函数：计算分割损失（修复：确保权重在正确设备上）
def compute_segmentation_loss(outputs, targets, weights, num_classes, dice_loss, focal_loss):
    """
    计算分割损失，支持dice_loss和focal_loss
    修复：确保权重张量与模型在同一设备上
    """
    # 修复1：将权重张量移到与outputs相同的设备上
    weight_tensor = torch.from_numpy(weights).float().to(outputs.device)

    # 基础损失：带权重的交叉熵损失
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    ce_loss = ce_loss_fn(outputs, targets)

    total_loss = ce_loss

    # 添加Dice损失（对类别不平衡更鲁棒）
    if dice_loss:
        dice_loss_value = 0
        smooth = 1e-5
        outputs_soft = torch.nn.functional.softmax(outputs, dim=1)

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
    class_counts = np.zeros(num_classes, dtype=np.float32)
    total_pixels = 0
    for line in tqdm(lines, desc="Calculating class weights"):
        name = line.split()[0]
        label_path = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass", name + ".png")
        label = np.array(Image.open(label_path))
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


def compute_class_weights(class_ratios, method='median_frequency'):
    if method == 'inverse':
        cls_weights = 1.0 / (class_ratios + 1e-8)
        cls_weights = cls_weights / cls_weights.min()
    elif method == 'median_frequency':
        median_frequency = np.median(class_ratios)
        cls_weights = median_frequency / (class_ratios + 1e-8)
        cls_weights = cls_weights / cls_weights.min()
    elif method == 'log':
        cls_weights = np.log(1.02 / (class_ratios + 0.02))
        cls_weights = cls_weights / cls_weights.min()
    elif method == 'custom':
        cls_weights = np.array([1, 80, 3], np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")
    print("\nComputed Class Weights:")
    for c in range(len(cls_weights)):
        print(f"Class {c}: {cls_weights[c]:.4f}")
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
        # 关键修复：写文件前强制创建目录（确保万无一失）
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "acc_epoch.txt"), 'a') as f:
            f.write(f"{epoch},{train_seg_acc},{val_seg_acc},{domain_acc},"
                    f"{domain_acc_backbone_source},{domain_acc_backbone_target},"
                    f"{domain_acc_aspp_source},{domain_acc_aspp_target}\n")

        self.acc_plot()

    def acc_plot(self):
        iters = range(len(self.acc_details["train_seg_acc"]))
        import scipy.signal

        plt.figure()
        plt.plot(iters, self.acc_details["train_seg_acc"], 'red', linewidth=2, label='train seg acc')
        plt.plot(iters, self.acc_details["val_seg_acc"], 'coral', linewidth=2, label='val seg acc')
        try:
            num = 5 if len(iters) < 25 else 15
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
            num = 5 if len(iters) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.acc_details["domain_acc"], num, 3), 'darkblue',
                     linestyle='--', linewidth=2, label='smooth domain acc')
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
        if num_classes == 3:
            self.colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
        else:
            hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                                   hsv_tuples))
            self.colors[0] = (0, 0, 0)
            # 手动设置颜色：索引0为背景（黑色），索引1为前景（浅蓝色，与标注匹配）
        self.colors = [(0, 0, 0), (173, 216, 230)]
        # 确保颜色数量与类别数一致
        assert len(self.colors) == num_classes, f"颜色数量{len(self.colors)}与类别数{num_classes}不匹配"
    def detect_image(self, model, image):
        iw, ih = image.size
        image_data, (pad_w, pad_h, scale, (new_w, new_h)) = self.resize_image_with_info(
            image, (self.input_shape[1], self.input_shape[0])
        )
        image_data_np = np.array(image_data, np.float32) / 255.0
        image_data_np = np.transpose(image_data_np, (2, 0, 1))
        image_data = np.expand_dims(image_data_np, 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if torch.cuda.is_available():
                images = images.cuda()
            outputs = model(images, mode='val')
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            pr = outputs.cpu().numpy()[0].transpose(1, 2, 0)
            pr = np.argmax(pr, axis=-1)

            if pad_h > 0 or pad_w > 0:
                valid_h = self.input_shape[0] - 2 * pad_h
                valid_w = self.input_shape[1] - 2 * pad_w
                pr = pr[pad_h:pad_h + valid_h, pad_w:pad_w + valid_w]

            pr = cv2.resize(pr.astype(np.float32), (iw, ih), interpolation=cv2.INTER_NEAREST).astype(np.int64)

        seg_img = np.zeros((pr.shape[0], pr.shape[1], 3), dtype=np.uint8)
        for c in range(self.num_classes):
            mask = (pr == c)
            seg_img[mask] = self.colors[c]
        return Image.fromarray(seg_img)

    def resize_image_with_info(self, image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        pad_w = (w - nw) // 2
        pad_h = (h - nh) // 2
        image_resized = image.resize((nw, nh), Image.BILINEAR)
        fill_color = (0, 0, 0)
        new_image = Image.new('RGB', (w, h), fill_color)
        new_image.paste(image_resized, (pad_w, pad_h))
        return new_image, (pad_w, pad_h, scale, (nw, nh))

    def visualize_results(self, model, lines, VOCdevkit_path, save_dir, num_visual=5, domain='source'):
        os.makedirs(save_dir, exist_ok=True)
        print(f"处理 {domain} 领域的可视化结果到 {save_dir}，数量: {min(num_visual, len(lines))}")
        indices = np.random.choice(len(lines), min(num_visual, len(lines)), replace=False)
        images = []
        names = []
        ground_truths = []

        print("批量读取图像数据...")
        for i in indices:
            line = lines[i]
            name = line.split()[0]
            names.append(name)
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages", name + ".jpg")
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            if domain == 'source':
                label_path = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass", name + ".png")
                label = Image.open(label_path)
                ground_truths.append(label)

        print("批量进行分割预测...")
        seg_imgs = [self.detect_image(model, img) for img in images]

        print("批量生成可视化结果...")
        for i, (name, image, seg_img) in enumerate(zip(names, images, seg_imgs)):
            if domain == 'source':
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(np.array(image))
                axes[0].set_title(f'Original Image: {name}')
                axes[0].axis('off')
                label_array = np.array(ground_truths[i])
                label_vis = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
                # 标注可视化也使用self.colors，确保与预测颜色一致
                for c in range(self.num_classes):
                    label_vis[label_array == c] = self.colors[c]
                axes[1].imshow(label_vis)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                axes[2].imshow(np.array(seg_img))
                axes[2].set_title('Prediction')
                axes[2].axis('off')
            else:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(np.array(image))
                axes[0].set_title(f'Original Image: {name}')
                axes[0].axis('off')
                axes[1].imshow(np.array(seg_img))
                axes[1].set_title('Prediction')
                axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}_comparison.jpg"),
                        dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            if (i + 1) % 10 == 0 or (i + 1) == len(indices):
                print(f"已处理 {i + 1}/{len(indices)} 张图像")
        print(f"可视化完成！结果保存到: {save_dir}")


# ==============================
# 5. 修改后的训练函数（核心：修复设备不匹配问题）
# ==============================
def fit_one_epoch_dann(model_train, model, loss_history, eval_callback, optimizer, epoch,
                       epoch_step, epoch_step_val, gen_source, gen_target, gen_val, UnFreeze_Epoch, Cuda,
                       dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                       save_period, save_dir, local_rank, lambda_domain=0.5):
    total_loss = 0
    total_seg_loss = 0
    total_domain_loss_backbone = 0
    total_domain_loss_aspp = 0
    total_domain_loss = 0
    val_loss = 0

    total_domain_acc_backbone_source = 0
    total_domain_acc_backbone_target = 0
    total_domain_acc_aspp_source = 0
    total_domain_acc_aspp_target = 0
    total_domain_acc = 0

    total_seg_acc = 0
    val_seg_acc = 0

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    class_iou = np.zeros(num_classes)

    p = float(epoch) / UnFreeze_Epoch
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    if hasattr(model_train, 'module'):
        model_train.module.set_alpha(alpha)
    else:
        model_train.set_alpha(alpha)

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    model_train = model_train.to('cuda')
    # 创建源域和目标域数据迭代器
    source_iter = iter(gen_source)
    target_iter = iter(gen_target)

    for iteration in range(epoch_step):
        try:
            # 关键修改1：解包时获取文件名（imgs, masks, labels, file_names）
            imgs_source, pngs_source, labels_source, file_names_source = next(source_iter)
        except StopIteration:
            source_iter = iter(gen_source)
            imgs_source, pngs_source, labels_source, file_names_source = next(source_iter)

        try:
            # 关键修改2：目标域同理获取文件名
            imgs_target, pngs_target, labels_target, file_names_target = next(target_iter)
        except StopIteration:
            target_iter = iter(gen_target)
            imgs_target, pngs_target, labels_target, file_names_target = next(target_iter)

        # 处理源域数据
        # 修复2：创建领域标签时指定设备（与数据一致）
        device = imgs_source.device  # 获取数据所在设备（CPU/CUDA）
        domain_labels_source = torch.zeros(imgs_source.size(0), dtype=torch.long).to(device)  # 源域标签为0

        # 通过文件名判断是否为混入的目标域样本
        for i, file_name in enumerate(file_names_source):
            base_filename = os.path.splitext(file_name)[0]
            if base_filename in TARGET_FILENAMES:
                domain_labels_source[i] = 1  # 混入的目标域样本设为1

        # 处理目标域数据
        # 修复3：目标域标签也指定设备
        domain_labels_target = torch.ones(imgs_target.size(0), dtype=torch.long).to(device)  # 目标域标签为1

        if Cuda:
            # 确保所有数据都在CUDA上（双重保险）
            imgs_source = imgs_source.cuda(non_blocking=True)
            pngs_source = pngs_source.cuda(non_blocking=True)
            domain_labels_source = domain_labels_source.cuda(non_blocking=True)

            imgs_target = imgs_target.cuda(non_blocking=True)
            domain_labels_target = domain_labels_target.cuda(non_blocking=True)
            model_train = model_train.to('cuda')
        with torch.cuda.amp.autocast(enabled=fp16):
            # 源域前向传播
            result = model_train(imgs_source, alpha=alpha, mode='train')
            seg_output_source, domain_output_backbone_source, domain_output_aspp_source = result

            # 目标域前向传播
            _, domain_output_backbone_target, domain_output_aspp_target = model_train(imgs_target, alpha=alpha,
                                                                                      mode='train')

            # 计算分割损失（仅源域）- 已在函数内修复设备问题
            seg_loss = compute_segmentation_loss(seg_output_source, pngs_source, weights=cls_weights,
                                                 num_classes=num_classes, dice_loss=dice_loss,
                                                 focal_loss=focal_loss)

            # 计算两个域分类器的损失（源域+目标域）
            domain_loss_backbone_source = torch.nn.functional.cross_entropy(domain_output_backbone_source,
                                                                            domain_labels_source)
            domain_loss_backbone_target = torch.nn.functional.cross_entropy(domain_output_backbone_target,
                                                                            domain_labels_target)
            domain_loss_backbone = domain_loss_backbone_source + domain_loss_backbone_target

            domain_loss_aspp_source = torch.nn.functional.cross_entropy(domain_output_aspp_source, domain_labels_source)
            domain_loss_aspp_target = torch.nn.functional.cross_entropy(domain_output_aspp_target, domain_labels_target)
            domain_loss_aspp = domain_loss_aspp_source + domain_loss_aspp_target

            domain_loss = (domain_loss_backbone + domain_loss_aspp) / 2
            loss = seg_loss + lambda_domain * domain_loss

            # 计算域分类准确率
            domain_pred_backbone_source = torch.argmax(domain_output_backbone_source, dim=1)
            domain_acc_backbone_source = (domain_pred_backbone_source == domain_labels_source).float().mean()

            domain_pred_backbone_target = torch.argmax(domain_output_backbone_target, dim=1)
            domain_acc_backbone_target = (domain_pred_backbone_target == domain_labels_target).float().mean()

            domain_pred_aspp_source = torch.argmax(domain_output_aspp_source, dim=1)
            domain_acc_aspp_source = (domain_pred_aspp_source == domain_labels_source).float().mean()

            domain_pred_aspp_target = torch.argmax(domain_output_aspp_target, dim=1)
            domain_acc_aspp_target = (domain_pred_aspp_target == domain_labels_target).float().mean()

            domain_acc = (domain_acc_backbone_source + domain_acc_backbone_target +
                          domain_acc_aspp_source + domain_acc_aspp_target) / 4

            # 计算分割准确率
            seg_pred = torch.argmax(seg_output_source, dim=1)
            seg_acc = (seg_pred == pngs_source).float()
            mask = pngs_source != 255
            seg_acc = seg_acc[mask].mean() if mask.any() else torch.tensor(0.0, device=seg_acc.device)

            # 计算每个类别的准确率和IoU
            for c in range(num_classes):
                mask_c = (pngs_source == c)
                if mask_c.any():
                    correct_c = (seg_pred[mask_c] == pngs_source[mask_c]).float().sum().item()
                    class_correct[c] += correct_c
                    class_total[c] += mask_c.sum().item()

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
        total_domain_loss_backbone += domain_loss_backbone.item()
        total_domain_loss_aspp += domain_loss_aspp.item()
        total_domain_loss += domain_loss.item()

        total_domain_acc_backbone_source += domain_acc_backbone_source.item()
        total_domain_acc_backbone_target += domain_acc_backbone_target.item()
        total_domain_acc_aspp_source += domain_acc_aspp_source.item()
        total_domain_acc_aspp_target += domain_acc_aspp_target.item()
        total_domain_acc += domain_acc.item()
        total_seg_acc += seg_acc.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'seg_loss': total_seg_loss / (iteration + 1),
                                'domain_loss_backbone': total_domain_loss_backbone / (iteration + 1),
                                'domain_loss_aspp': total_domain_loss_aspp / (iteration + 1),
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
        # 验证集同理解包文件名（如需使用）
        imgs, pngs, labels, val_file_names = batch
        with torch.no_grad():
            if Cuda:
                imgs = imgs.cuda(non_blocking=True)
                pngs = pngs.cuda(non_blocking=True)

            outputs = model_train(imgs, mode='val')
            loss = compute_segmentation_loss(outputs, pngs, weights=cls_weights,
                                             num_classes=num_classes, dice_loss=dice_loss,
                                             focal_loss=focal_loss)
            val_loss += loss.item()

            seg_pred = torch.argmax(outputs, dim=1)
            seg_acc = (seg_pred == pngs).float()
            mask = pngs != 255
            seg_acc = seg_acc[mask].mean() if mask.any() else torch.tensor(0.0, device=seg_acc.device)
            val_seg_acc += seg_acc.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'val_seg_acc': val_seg_acc / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
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
        print('Domain Acc: %.3f%% (Backbone Source: %.3f%%, Target: %.3f%%)' %
              (total_domain_acc / epoch_step * 100,
               total_domain_acc_backbone_source / epoch_step * 100,
               total_domain_acc_backbone_target / epoch_step * 100))
        print('Domain Acc: %.3f%% (ASPP Source: %.3f%%, Target: %.3f%%)' %
              (total_domain_acc / epoch_step * 100,
               total_domain_acc_aspp_source / epoch_step * 100,
               total_domain_acc_aspp_target / epoch_step * 100))

        print("\nClass-wise Performance:")
        for c in range(num_classes):
            if class_total[c] > 0:
                acc = class_correct[c] / class_total[c] * 100
                iou = class_iou[c] / epoch_step * 100
                print(f"Class {c}: Acc={acc:.2f}%, IoU={iou:.2f}%")
            else:
                print(f"Class {c}: No samples")

        # 确保保存目录存在
        os.makedirs(os.path.join(save_dir, 'pth'), exist_ok=True)
        if (epoch + 1) % save_period == 0 or epoch + 1 == UnFreeze_Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'pth/ep%03d-loss%.3f-val_loss%.3f.pth' %
                                                        (
                                                            epoch + 1, total_loss / epoch_step,
                                                            val_loss / epoch_step_val)))

        if eval_callback:
            eval_callback.on_epoch_end(epoch + 1, model_train)

    return total_loss / epoch_step, val_loss / epoch_step_val


# ==============================
# 6. 主函数（保持不变，添加CUDA可用性检查）
# ==============================
if __name__ == "__main__":
    # 基本配置参数
    Cuda = torch.cuda.is_available()  # 修复：自动检测CUDA可用性
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = False
    num_classes = 2
    backbone = "ghostnet"
    pretrained = False
    model_path = ""
    downsample_factor = 16
    input_shape = [640, 640]

    # DANN特定参数
    use_dann = True
    lambda_domain = 0.5

    # 数据集路径配置
    source_VOCdevkit_path = 'F:\BaiduNetdiskDownload\VOCdevkit_1-2仁'
    target_VOCdevkit_path = 'H:\板栗\VOCdevkit'

    # 训练参数
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 2
    UnFreeze_Epoch = 500
    Unfreeze_batch_size = 2
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
    num_workers = 4

    # 设置随机种子

    # 设置显卡
    ngpus_per_node = torch.cuda.device_count() if Cuda else 0
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if Cuda else 'cpu')
        local_rank = 0
        rank = 0

    # 下载预训练权重
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    # 创建模型
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

    # 加载预训练权重
    if not pretrained:
        weights_init(model)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)  # 修复：加载权重时指定设备

        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                if k.startswith('deeplab.'):
                    new_k = k[8:]
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

    # 记录Loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        os.makedirs(log_dir, exist_ok=True)  # 自动创建日志目录（不存在则新建）
        loss_history = EnhancedLossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # FP16设置
    if fp16 and Cuda:  # 修复：只有CUDA可用时才启用FP16
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None
        fp16 = False  # 强制关闭FP16

    model_train = model.train()

    # 多卡同步Bn
    if sync_bn and ngpus_per_node > 1 and distributed and Cuda:  # 修复：添加CUDA检查
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed or no CUDA.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # 读取数据集对应的txt
    # 修复后（过滤空行）：
    with open(os.path.join(source_VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        source_train_lines = [line.strip() for line in f.readlines() if line.strip()]  # 只保留非空行
    with open(os.path.join(source_VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        source_val_lines = [line.strip() for line in f.readlines() if line.strip()]

    with open(os.path.join(target_VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        target_train_lines = [line.strip() for line in f.readlines() if line.strip()]

    num_train = len(source_train_lines)
    num_val = len(source_val_lines)

    # 计算类别权重
    if local_rank == 0:
        print("\nCalculating class weights for source domain...")
        cls_weights = [1, 1]
        print("\nFinal Class Weights:")
        for c in range(num_classes):
            print(f"Class {c}: {cls_weights[c]:.4f}")
        cls_weights = np.ones([num_classes], np.float32)
    else:
        cls_weights = np.ones([num_classes], np.float32)

    # 广播类别权重
    if distributed and Cuda:  # 修复：只有分布式且CUDA可用时才广播
        cls_weights = torch.from_numpy(cls_weights).float().cuda(local_rank)
        dist.broadcast(cls_weights, src=0)
        cls_weights = cls_weights.cpu().numpy()

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
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

    # 开始训练
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        if use_dann:
            params_to_optimize = model_train.parameters()
        else:
            params_to_optimize = model_train.parameters()

        optimizer = {
            'adam': optim.AdamW(params_to_optimize, Init_lr_fit, betas=(momentum, 0.9), weight_decay=weight_decay),
        }[optimizer_type]

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # 创建数据集（使用修改后的DeeplabDataset）
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

        # 创建数据加载器（使用修改后的collate_fn）
        gen_source = DataLoader(source_train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=Cuda,  # 修复：只有CUDA可用时才启用pin_memory
                                drop_last=True, collate_fn=deeplab_dataset_collate,
                                sampler=source_train_sampler,
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        gen_target = DataLoader(target_train_dataset, shuffle=shuffle, batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=Cuda,  # 修复：只有CUDA可用时才启用pin_memory
                                drop_last=True, collate_fn=deeplab_dataset_collate,
                                sampler=target_train_sampler,
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=Cuda,  # 修复：只有CUDA可用时才启用pin_memory
                             drop_last=True, collate_fn=deeplab_dataset_collate,
                             sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # 记录eval的map曲线
        if local_rank == 0:
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

        # 开始模型训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 解冻逻辑
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 16
                lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
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
                                        pin_memory=Cuda,  # 修复：pin_memory适配CUDA
                                        drop_last=True, collate_fn=deeplab_dataset_collate,
                                        sampler=source_train_sampler,
                                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                gen_target = DataLoader(target_train_dataset, shuffle=shuffle, batch_size=batch_size,
                                        num_workers=num_workers,
                                        pin_memory=Cuda,  # 修复：pin_memory适配CUDA
                                        drop_last=True, collate_fn=deeplab_dataset_collate,
                                        sampler=target_train_sampler,
                                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=Cuda,  # 修复：pin_memory适配CUDA
                                     drop_last=True, collate_fn=deeplab_dataset_collate,
                                     sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                if source_train_sampler is not None:
                    source_train_sampler.set_epoch(epoch)
                if target_train_sampler is not None:
                    target_train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # 使用DANN训练函数
            if use_dann:
                fit_one_epoch_dann(model_train, model, loss_history, eval_callback, optimizer, epoch,
                                   epoch_step, epoch_step_val, gen_source, gen_target, gen_val, UnFreeze_Epoch, Cuda,
                                   dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                                   save_period, save_dir, local_rank, lambda_domain=lambda_domain)
            else:
                print("原始训练函数需要您根据原始代码提供")
                break

            # 可视化训练结果
            if local_rank == 0:
                epoch_train_visual_dir = os.path.join(train_visual_dir, f"epoch_{epoch + 1}")
                os.makedirs(epoch_train_visual_dir, exist_ok=True)

                source_visual_dir = os.path.join(epoch_train_visual_dir, "source")
                target_visual_dir = os.path.join(epoch_train_visual_dir, "target")
                os.makedirs(source_visual_dir, exist_ok=True)
                os.makedirs(target_visual_dir, exist_ok=True)

                visual_tool.visualize_results(
                    model_train,
                    source_train_lines,
                    source_VOCdevkit_path,
                    source_visual_dir,
                    20,
                    domain='source'
                )

                visual_tool.visualize_results(
                    model_train,
                    target_train_lines,
                    target_VOCdevkit_path,
                    target_visual_dir,
                    50,
                    domain='target'
                )

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
