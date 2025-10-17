# ----------------------------------------------------#
#   基于DANN训练结果的预测程序
#   支持单张图片预测、摄像头检测、文件夹预测和FPS测试
#   增加了类别判断和统计信息显示功能
# ----------------------------------------------------#
import datetime
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入DANN模型相关定义
from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import weights_init



# DANN版本的DeepLabV3+模型（预测版本）
class DeepLabDANNPredict(torch.nn.Module):
    def __init__(self, num_classes, backbone, downsample_factor, pretrained=False):
        super(DeepLabDANNPredict, self).__init__()
        # 原始DeepLabV3+模型
        self.deeplab = DeepLab(num_classes=num_classes, backbone=backbone,
                               downsample_factor=downsample_factor, pretrained=pretrained)

    def forward(self, x):
        # 获取输入图像尺寸
        H, W = x.size(2), x.size(3)

        # 获取DeepLabV3+的特征
        low_level_features, x_backbone = self.deeplab.backbone(x)
        x_aspp = self.deeplab.aspp(x_backbone)

        # 继续分割解码过程
        low_level_features = self.deeplab.shortcut_conv(low_level_features)
        x_aspp = torch.nn.functional.interpolate(x_aspp, size=(low_level_features.size(2), low_level_features.size(3)),
                                                 mode='bilinear', align_corners=True)
        x = self.deeplab.cat_conv(torch.cat((x_aspp, low_level_features), dim=1))
        x = self.deeplab.cls_conv(x)
        x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x


# 主要的预测类
class DeeplabDANNPredictor:
    def __init__(self, model_path, num_classes=3, backbone="ghostnet", downsample_factor=16, input_shape=[320, 320]):
        # 初始化模型参数
        self.num_classes = num_classes
        self.backbone = backbone
        self.downsample_factor = downsample_factor
        self.input_shape = input_shape

        # 创建模型（预测时不需要域分类器）
        self.model = DeepLabDANNPredict(
            num_classes=num_classes,
            backbone=backbone,
            downsample_factor=downsample_factor,
            pretrained=False
        )

        # 加载预训练权重
        if model_path != '':
            print('Loading weights from {}'.format(model_path))
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(model_path, map_location='cpu')

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
            self.model.load_state_dict(model_dict, strict=False)

            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            if no_load_key:
                print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n模型加载完成!")

        # 使用GPU（如果可用）
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model = self.model.cuda()

        self.model.eval()

        print('{} model loaded.'.format(model_path))

        # 设置颜色（可根据需要修改）
        if num_classes == 3:
            self.colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]  # 背景、类别1、类别2
        else:
            # 生成随机颜色
            hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                                   hsv_tuples))
            self.colors[0] = (0, 0, 0)  # 背景设为黑色

    def detect_image(self, image, count=False, name_classes=None):
        # 转换图像为RGB（如果必要）
        image = image.convert('RGB')

        # 获取原始图像尺寸
        iw, ih = image.size
        old_size = (ih, iw)  # (height, width)

        # 记录原始图像
        original_image = np.array(image)

        # 调整图像尺寸 - 使用与训练完全一致的方法
        image_data, (pad_w, pad_h, scale, (new_w, new_h)) = self.resize_image_with_info(
            image, (self.input_shape[1], self.input_shape[0])
        )

        # 添加批次维度并转换为tensor
        image_data_np = np.array(image_data, np.float32) / 255.0
        image_data_np = np.transpose(image_data_np, (2, 0, 1))
        image_data = np.expand_dims(image_data_np, 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 模型预测
            outputs = self.model(images)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            pr = outputs.cpu().numpy()

            # 获取预测结果
            pr = pr[0]
            pr = pr.transpose(1, 2, 0)
            pr = np.argmax(pr, axis=-1)

            # 关键修改：正确处理填充区域的掩码
            # 1. 先裁剪掉填充部分，得到有效区域的预测
            if pad_h > 0 or pad_w > 0:
                # 计算有效区域（去除填充）
                valid_h = self.input_shape[0] - 2 * pad_h
                valid_w = self.input_shape[1] - 2 * pad_w
                pr = pr[pad_h:pad_h + valid_h, pad_w:pad_w + valid_w]

            # 2. 将有效区域的预测缩放到原始图像尺寸
            pr = cv2.resize(pr.astype(np.float32), (iw, ih),
                            interpolation=cv2.INTER_NEAREST)
            pr = pr.astype(np.int64)

        # 计算每个类别的像素数量和比例
        classes_nums = np.zeros([self.num_classes])
        total_points_num = old_size[0] * old_size[1]
        class_ratios = []

        for i in range(self.num_classes):
            num = np.sum(pr == i)
            ratio = num / total_points_num * 100
            classes_nums[i] = num
            class_ratios.append(ratio)

        # 打印统计信息（如果count=True）
        if count:
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                if classes_nums[i] > 0:
                    print(
                        "|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(int(classes_nums[i])), class_ratios[i]))
                    print('-' * 63)
            print("sum px:", total_points_num)
            print(f"原始图像尺寸: {old_size}")
            print(f"模型输入尺寸: {self.input_shape}")
            print(f"缩放比例: {scale:.4f}")
            print(f"填充尺寸: ({pad_w}, {pad_h})")

        # 生成分割图像 - 使用更明显的颜色
        seg_img = np.zeros((pr.shape[0], pr.shape[1], 3), dtype=np.uint8)
        for c in range(self.num_classes):
            mask = (pr == c)
            seg_img[mask] = self.colors[c]
            if c==2:
                print(pr == c)

        # 将分割图像与原始图像混合
        image_array = original_image.copy()

        # 使用更明显的混合比例，便于检查对齐情况
        alpha = 0.7  # 增加分割结果的透明度，便于检查对齐
        result_image = cv2.addWeighted(image_array, 1 - alpha, seg_img, alpha, 0)

        # 添加边界框显示有效区域，便于调试
        if count:
            # 在图像上绘制有效区域边界
            cv2.rectangle(result_image, (0, 0), (iw - 1, ih - 1), (0, 255, 0), 2)
            # 添加尺寸信息文本
            cv2.putText(result_image, f'Original: {iw}x{ih}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_image, f'Model input: {self.input_shape[1]}x{self.input_shape[0]}',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 转换为PIL图像以便添加文本
        image_pil = Image.fromarray(result_image)

        # 添加类别信息和统计结果到图像上
        if name_classes is not None:
            # 创建绘图对象
            draw = ImageDraw.Draw(image_pil)

            # 尝试加载更大更粗的字体
            try:
                # 尝试加载Arial Black或类似粗体字体
                font = ImageFont.truetype("arialbd.ttf", 64)  # 增大字体大小到28
            except:
                try:
                    font = ImageFont.truetype("Arial Bold.ttf", 64)
                except:
                    try:
                        # 尝试普通Arial但更大字号
                        font = ImageFont.truetype("arial.ttf", 64)
                    except:
                        try:
                            font = ImageFont.truetype("Arial.ttf", 64)
                        except:
                            # 最后回退到默认字体但增大尺寸
                            font = ImageFont.load_default()
                            # 对于默认字体，我们可能需要手动调整大小
                            font.size = 64  # 尝试设置大小属性

            # 确定主要类别（比例最高的非背景类别）
            main_class_idx = 0
            max_ratio = 0
            for i in range(1, self.num_classes):  # 从1开始，跳过背景
                if class_ratios[i] > max_ratio:
                    max_ratio = class_ratios[i]
                    main_class_idx = i

            # 增强文本可读性 - 使用更醒目的背景和文字颜色
            def draw_text_with_bg(draw, text, position, font, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
                """绘制带背景框的文本，增强可读性"""
                # 计算文本边界框
                bbox = draw.textbbox(position, text, font=font)

                # 扩展背景框，增加内边距
                expanded_bbox = (
                    bbox[0] - 8,  # 左
                    bbox[1] - 4,  # 上
                    bbox[2] + 8,  # 右
                    bbox[3] + 4  # 下
                )

                # 绘制半透明背景
                bg_layer = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
                bg_draw = ImageDraw.Draw(bg_layer)
                bg_draw.rectangle(expanded_bbox, fill=(bg_color[0], bg_color[1], bg_color[2], 220))  # 220/255透明度
                image_pil.paste(Image.alpha_composite(image_pil.convert('RGBA'), bg_layer))

                # 重新创建绘图对象
                draw = ImageDraw.Draw(image_pil)

                # 绘制文本
                draw.text(position, text, font=font, fill=text_color)

                return expanded_bbox[3] - expanded_bbox[1] + 8  # 返回高度用于下一行定位

            # 选择更醒目的颜色组合
            text_color = (255, 255, 255)  # 白色文字
            bg_color = (0, 0, 0)  # 黑色背景

            y_offset = 15
            # 添加主要类别信息 - 使用更醒目的颜色
            if max_ratio > 0:  # 确保有检测到非背景类别
                main_class_text = f"main_class: {name_classes[main_class_idx]} ({max_ratio:.2f}%)"
                text_height = draw_text_with_bg(draw, main_class_text, (15, y_offset), font,
                                                text_color, bg_color)
                y_offset += text_height

            # 添加背景类别信息
            if classes_nums[0] > 0:
                bg_text = f"background: {class_ratios[0]:.2f}%"
                text_height = draw_text_with_bg(draw, bg_text, (15, y_offset), font,
                                                text_color, bg_color)
                y_offset += text_height

            # 添加其他类别信息
            for i in range(1, self.num_classes):
                if classes_nums[i] > 0 and i != main_class_idx:
                    class_text = f"{name_classes[i]}: {class_ratios[i]:.2f}%"
                    text_height = draw_text_with_bg(draw, class_text, (15, y_offset), font,
                                                    text_color, bg_color)
                    y_offset += text_height

            # 添加总像素数 - 放在右下角避免遮挡重要区域
            sum_text = f"sum_px: {total_points_num}"
            text_bbox = draw.textbbox((0, 0), sum_text, font=font)
            text_width = text_bbox[2] - text_bbox[0] + 30
            draw_text_with_bg(draw, sum_text, (iw - text_width, ih - 50), font,
                              text_color, bg_color)

        return image_pil

    def resize_image(self, image, size):
        """保持与之前兼容的接口"""
        resized, _ = self.resize_image_with_info(image, size)
        return resized

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
        image_resized = image.resize((nw, nh), Image.BILINEAR)  # 改为BILINEAR，与训练一致

        # 使用与训练时相同的填充颜色（通常是0或均值）
        # 重要：这里应该与训练时的填充颜色一致！
        fill_color = (0, 0, 0)  # 改为黑色填充，或者使用训练时的设置
        new_image = Image.new('RGB', (w, h), fill_color)
        new_image.paste(image_resized, (pad_w, pad_h))

        return new_image, (pad_w, pad_h, scale, (nw, nh))

    # 新增调试函数
    def debug_detect_image(self, image, name_classes=None):
        """调试版本，显示详细的变换信息"""
        return self.detect_image(image, count=True, name_classes=name_classes)

    # 新增函数：检查训练和推理的一致性
    def check_preprocessing_consistency(self, image_path):
        """
        检查训练和推理预处理是否一致
        """
        image = Image.open(image_path)
        iw, ih = image.size

        print("=== 预处理一致性检查 ===")
        print(f"原始图像尺寸: {iw}x{ih}")
        print(f"模型输入尺寸: {self.input_shape[1]}x{self.input_shape[0]}")

        # 模拟训练时的预处理
        image_data = self.resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_array = np.array(image_data)

        print(f"预处理后尺寸: {image_array.shape}")
        print(f"填充颜色: {image_array[0, 0]} (左上角像素)")
        print(f"图像均值: {np.mean(image_array, axis=(0, 1))}")

        return image_data

    # def get_FPS(self, image, test_interval):
    #     # 调整图像尺寸
    #     image_data = self.resize_image(image, (self.input_shape[1], self.input_shape[0]))
    #
    #     # 添加批次维度并转换为tensor
    #     image_data = np.expand_dims(np.transpose(np.array(image_data, np.float32) / 255.0, (2, 0, 1)), 0)
    #
    #     with torch.no_grad():
    #         images = torch.from_numpy(image_data)
    #         if self.cuda:
    #             images = images.cuda()
    #
    #         # 预热
    #         for _ in range(10):
    #             outputs = self.model(images)
    #
    #         # 测试时间
    #         time1 = time.time()
    #         for _ in range(test_interval):
    #             outputs = self.model(images)
    #         time2 = time.time()
    #
    #         tact_time = (time2 - time1) / test_interval
    #         return tact_time


if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   基本配置参数（需要根据您的训练设置进行调整）
    # -------------------------------------------------------------------------#
    num_classes = 3  # 类别数（包括背景）
    backbone = "ghostnet"  # backbone网络
    downsample_factor = 16  # 下采样因子
    input_shape = [320, 320]  # 输入图像尺寸

    # -------------------------------------------------------------------------#
    #   模型路径（请修改为您训练好的模型路径）
    # -------------------------------------------------------------------------#
    model_path = "logs/ep005-loss0.795-val_loss0.051.pth"  # 替换为您的模型路径

    # -------------------------------------------------------------------------#
    #   创建预测器
    # -------------------------------------------------------------------------#
    deeplab = DeeplabDANNPredictor(
        model_path=model_path,
        num_classes=num_classes,
        backbone=backbone,
        downsample_factor=downsample_factor,
        input_shape=input_shape
    )

    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测
    #   'video'             表示视频检测
    #   'fps'               表示测试fps
    #   'dir_predict'       表示遍历文件夹进行检测并保存
    # ----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"

    # -------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和训练时一样
    # -------------------------------------------------------------------------#
    count = False
    name_classes = ["_background_", "ellipse", "ellipse_half"]  # 根据您的类别修改

    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps           用于保存的视频的fps
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0

    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数
    #   fps_image_path      用于指定测试的fps图片
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"

    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    # -------------------------------------------------------------------------#
    # dir_origin_path = 'F:\\BaiduNetdiskDownload\\板栗\\archive\\chestnut_zonguldak'
    # dir_origin_path = 'F:\BaiduNetdiskDownload\胡桃仁-3-23\\all'
    dir_origin_path = 'F:\BaiduNetdiskDownload\胡桃仁-3-23\二路'
    dir_save_path = "img_out/"

    if mode == "predict":
        '''
        predict模式有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(deeplab.detect_image(frame, name_classes=name_classes))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = deeplab.detect_image(image, name_classes=name_classes)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")