import os
from PIL import Image


def center_crop_image(input_path, output_path, target_width=1024, target_height=768):
    """
    将单张图片以中心点裁剪为指定尺寸

    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        target_width: 目标宽度，默认1024
        target_height: 目标高度，默认768
    """
    try:
        # 打开图片文件[6](@ref)
        image = Image.open(input_path)

        # 获取图片的原始尺寸[4](@ref)
        width, height = image.size

        # 计算裁剪区域的坐标[3,4](@ref)
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2

        # 执行裁剪操作[6](@ref)
        cropped_image = image.crop((left, top, right, bottom))

        # 保存裁剪后的图片[4](@ref)
        cropped_image.save(output_path)
        print(f"成功裁剪: {os.path.basename(input_path)}")

    except Exception as e:
        print(f"处理图片 {input_path} 时出错: {str(e)}")


def batch_center_crop_images(input_folder, output_folder, target_width=1024, target_height=768):
    """
    批量裁剪文件夹中的所有图片

    Args:
        input_folder: 输入图片文件夹路径
        output_folder: 输出图片文件夹路径
        target_width: 目标宽度，默认1024
        target_height: 目标高度，默认768
    """
    # 创建输出文件夹（如果不存在）[5](@ref)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 支持的图片格式[7](@ref)
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # 统计处理结果
    processed_count = 0
    error_count = 0

    # 遍历输入文件夹中的所有文件[5,7](@ref)
    for filename in os.listdir(input_folder):
        # 检查文件格式
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in supported_formats:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                center_crop_image(input_path, output_path, target_width, target_height)
                processed_count += 1
            except Exception as e:
                print(f"处理失败: {filename} - 错误: {str(e)}")
                error_count += 1
        else:
            print(f"跳过不支持的文件: {filename}")

    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 张图片")
    print(f"处理失败: {error_count} 张图片")
    print(f"输出位置: {output_folder}")


if __name__ == "__main__":
    # 配置参数
    input_folder = "H:\板栗\VOCdevkit\VOC2007\JPEGImages"  # 替换为你的输入文件夹路径
    output_folder = "H:\板栗\VOCdevkit\\1600_1024"  # 替换为你的输出文件夹路径
    target_width = 1600
    target_height = 1024

    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在！")
        print("请确保指定的输入文件夹路径正确。")
    else:
        # 执行批量裁剪
        batch_center_crop_images(input_folder, output_folder, target_width, target_height)