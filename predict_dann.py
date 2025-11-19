import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nets.deeplabv3_plus import DeepLab

# 解决OpenMP冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------------------#
#   预测核心配置（需根据实际情况修改）
# ---------------------------#
class Config:
    model_path = "logs/半监督_512_468_32张/pth/ep140-loss0.704-val_loss0.008.pth"  # pth模型路径
    num_classes = 2  # 类别数（含背景）
    backbone = "ghostnet"  # 骨干网络（需与训练时一致）
    downsample_factor = 16  # 下采样因子（需与训练时一致）
    input_shape = [512, 468]  # 输入图像尺寸（需与训练时一致）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 运行设备


# ---------------------------#
#   保留模型核心结构（仅用于加载权重和预测）
# ---------------------------#
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainClassifier(torch.nn.Module):
    def __init__(self, input_channels=128, hidden_size=1024, num_domains=2):
        super(DomainClassifier, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_channels, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_size, num_domains),
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
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DeepLabDANN(torch.nn.Module):
    def __init__(self, num_classes, backbone, downsample_factor, pretrained=False, lambda_domain=0.5):
        super(DeepLabDANN, self).__init__()
        self.deeplab = DeepLab(num_classes=num_classes, backbone=backbone,
                               downsample_factor=downsample_factor, pretrained=pretrained)
        if backbone == 'ghostnet':
            backbone_output_channels = 96
        self.domain_classifier_backbone = DomainClassifier(backbone_output_channels)
        aspp_output_channels = 128
        low_level_features_channels = 12
        self.domain_classifier_aspp = DomainClassifier(low_level_features_channels)
        self.lambda_domain = lambda_domain
        self.alpha = 0

    def forward(self, x, alpha=None, mode='val'):
        if alpha is None:
            alpha = self.alpha
        H, W = x.size(2), x.size(3)
        low_level_features_backbone, x_backbone = self.deeplab.backbone(x)
        x_aspp = self.deeplab.aspp(x_backbone)
        low_level_features = self.deeplab.shortcut_conv(low_level_features_backbone)
        x_aspp = F.interpolate(x_aspp, size=(low_level_features.size(2), low_level_features.size(3)),
                               mode='bilinear', align_corners=True)
        cls_conv_before = self.deeplab.cat_conv(torch.cat((x_aspp, low_level_features), dim=1))
        x = self.deeplab.cls_conv(cls_conv_before)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x  # 仅返回分割结果，忽略域分类器输出

    def set_alpha(self, alpha):
        self.alpha = alpha


# ---------------------------#
#   图像预处理与后处理工具
# ---------------------------#
class SegmentationPredictor:
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()
        self.model.eval()  # 切换到评估模式

    def _load_model(self):
        """加载pth模型权重（修复权重匹配逻辑）"""
        model = DeepLabDANN(
            num_classes=self.config.num_classes,
            backbone=self.config.backbone,
            downsample_factor=self.config.downsample_factor,
            pretrained=False
        )

        # 加载权重
        print(f"正在加载模型：{self.config.model_path}")
        pretrained_dict = torch.load(self.config.model_path, map_location=self.config.device)
        model_dict = model.state_dict()

        # 修复：正确匹配权重键（兼容带/不带'deeplab.'前缀）
        load_dict = {}
        loaded_keys = []
        unloaded_keys = []

        for weight_key, weight_val in pretrained_dict.items():
            # 情况1：权重键直接匹配模型键
            if weight_key in model_dict and model_dict[weight_key].shape == weight_val.shape:
                load_dict[weight_key] = weight_val
                loaded_keys.append(weight_key)
            # 情况2：权重键带'deeplab.'前缀，去掉前缀后匹配
            elif weight_key.startswith('deeplab.'):
                model_key = weight_key[8:]  # 去掉'deeplab.'前缀
                if model_key in model_dict and model_dict[model_key].shape == weight_val.shape:
                    load_dict[model_key] = weight_val
                    loaded_keys.append(f"{weight_key} -> {model_key}")
            # 情况3：模型键带'deeplab.'前缀，权重键不带（反向兼容）
            elif f'deeplab.{weight_key}' in model_dict and model_dict[
                f'deeplab.{weight_key}'].shape == weight_val.shape:
                load_dict[f'deeplab.{weight_key}'] = weight_val
                loaded_keys.append(f"{weight_key} -> deeplab.{weight_key}")
            else:
                unloaded_keys.append(weight_key)

        # 更新模型权重
        model_dict.update(load_dict)
        model.load_state_dict(model_dict, strict=False)  # strict=False允许部分权重不匹配（如域分类器）
        model.to(self.config.device)

        # 打印权重加载日志
        print(f"模型加载完成，运行设备：{self.config.device}")
        print(f"成功加载权重数：{len(loaded_keys)}")
        for key in loaded_keys[:10]:  # 打印前10个成功加载的键
            print(f"  - {key}")
        if len(loaded_keys) > 10:
            print(f"  ... 还有 {len(loaded_keys) - 10} 个权重键")

        if unloaded_keys:
            print(f"\n未加载的权重数：{len(unloaded_keys)}（多为域分类器权重，可忽略）")
            for key in unloaded_keys[:5]:  # 打印前5个未加载的键
                print(f"  - {key}")
            if len(unloaded_keys) > 5:
                print(f"  ... 还有 {len(unloaded_keys) - 5} 个权重键")

        return model

    def _preprocess_image(self, image):
        """图像预处理：调整尺寸、归一化、维度转换"""
        iw, ih = image.size
        target_w, target_h = self.config.input_shape[1], self.config.input_shape[0]

        # 保持长宽比缩放
        scale = min(target_w / iw, target_h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        # 计算填充
        pad_w = (target_w - nw) // 2
        pad_h = (target_h - nh) // 2

        # 缩放并填充
        image_resized = image.resize((nw, nh), Image.BILINEAR)
        new_image = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        new_image.paste(image_resized, (pad_w, pad_h))

        # 归一化和维度转换
        image_data = np.array(new_image, np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # HWC -> CHW
        image_data = np.expand_dims(image_data, 0)  # 添加batch维度
        return torch.from_numpy(image_data).to(self.config.device), (iw, ih, pad_w, pad_h, scale)

    def _postprocess_mask(self, output, iw, ih, pad_w, pad_h, scale):
        """后处理：获取掩码并恢复原始尺寸"""
        # 取概率最大的类别
        output = F.softmax(output, dim=1)
        mask = torch.argmax(output, dim=1).cpu().numpy()[0]  # (H, W)

        # 去除填充区域
        valid_h = self.config.input_shape[0] - 2 * pad_h
        valid_w = self.config.input_shape[1] - 2 * pad_w
        mask = mask[pad_h:pad_h + valid_h, pad_w:pad_w + valid_w]

        # 恢复到原始图像尺寸
        mask = cv2.resize(mask.astype(np.float32), (iw, ih), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.int64)  # 最终掩码（值为类别索引）

    def predict(self, image_path):
        """单张图像预测：输入图像路径，输出分割掩码"""
        # 读取图像
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在：{image_path}")
        image = Image.open(image_path).convert('RGB')

        # 预处理
        input_tensor, (iw, ih, pad_w, pad_h, scale) = self._preprocess_image(image)

        # 预测（禁用梯度计算）
        with torch.no_grad():
            output = self.model(input_tensor)

        # 后处理得到掩码
        mask = self._postprocess_mask(output, iw, ih, pad_w, pad_h, scale)
        return mask

    def batch_predict(self, image_dir, save_dir=None):
        """批量预测：输入图像目录，可选保存掩码"""
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"图像目录不存在：{image_dir}")

        os.makedirs(save_dir, exist_ok=True) if save_dir else None
        # 获取所有图像路径（支持jpg、png、jpeg格式）
        image_ext = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                       if f.endswith(image_ext)]

        if not image_paths:
            print("警告：图像目录中未找到有效图像文件")
            return []

        masks = []
        for idx, img_path in enumerate(image_paths, 1):
            try:
                mask = self.predict(img_path)
                masks.append(mask)

                # 保存掩码（可选）
                if save_dir:
                    img_name = os.path.basename(img_path).split('.')[0]
                    # 保存为可视化彩色图像（非背景统一红色）
                    vis_mask = self._visualize_mask(mask)
                    cv2.imwrite(os.path.join(save_dir, f"{img_name}_mask_vis.png"), vis_mask)

                print(f"[{idx}/{len(image_paths)}] 处理完成：{os.path.basename(img_path)}")
            except Exception as e:
                print(f"[{idx}/{len(image_paths)}] 处理失败：{os.path.basename(img_path)}，错误：{str(e)}")

        print(f"\n批量预测完成，共处理 {len(image_paths)} 张图像，成功 {len(masks)} 张")
        if save_dir:
            print(f"预测结果已保存到：{save_dir}")
        return masks

    def _visualize_mask(self, mask):
        """掩码可视化：背景黑色，所有非背景类别统一红色"""
        vis_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # 背景（类别0）：黑色
        # 非背景（类别≥1）：红色 (255, 0, 0)
        vis_mask[mask != 0] = (0, 0, 255)

        return vis_mask


# ---------------------------#
#   主函数：快速使用示例
# ---------------------------#
if __name__ == "__main__":
    # 初始化配置
    config = Config()

    # 创建预测器（修复后不会再报KeyError）
    try:
        predictor = SegmentationPredictor(config)
    except Exception as e:
        print(f"创建预测器失败：{str(e)}")
        exit(1)

    # # 1. 单张图像预测（请替换为你的测试图像路径）
    # single_image_path = "test_image.jpg"  # 示例路径
    # if os.path.exists(single_image_path):
    #     try:
    #         mask = predictor.predict(single_image_path)
    #         print(f"\n单张图像预测完成：")
    #         print(f"  - 图像路径：{single_image_path}")
    #         print(f"  - 掩码形状：{mask.shape}")
    #         print(f"  - 类别范围：0-{config.num_classes - 1}")
    #         print(f"  - 各类别像素数：")
    #         for c in range(config.num_classes):
    #             print(f"    类别{c}：{np.sum(mask == c)} 个像素")
    #
    #         # 保存结果
    #         np.save("single_mask.npy", mask)
    #         vis_mask = predictor._visualize_mask(mask)
    #         cv2.imwrite("single_mask_vis.png", vis_mask)
    #         print(f"  - 结果保存：single_mask.npy（原始掩码）、single_mask_vis.png（可视化）")
    #     except Exception as e:
    #         print(f"单张图像预测失败：{str(e)}")

    # 2. 批量图像预测（已启用）
    batch_image_dir = r"D:\code\python\板栗拍照\photos_28_32"  # 修复路径转义问题
    save_mask_dir = "output_masks"  # 结果保存目录
    if os.path.exists(batch_image_dir):
        predictor.batch_predict(batch_image_dir, save_mask_dir)