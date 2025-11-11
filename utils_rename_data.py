import os


def batch_rename_images(folder_path, start_index=1, prefix="", digits=3):
    """
    批量重命名文件夹中的图片文件，按照数字顺序排列

    参数:
    folder_path: 图片文件夹路径
    start_index: 起始序号，默认为1
    prefix: 文件名前缀，默认为空
    digits: 序号位数（如3表示001, 002），默认为3
    """
    try:
        # 获取文件夹中的所有文件[1](@ref)
        all_files = os.listdir(folder_path)

        # 过滤出图片文件[1,2](@ref)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']
        image_files = [f for f in all_files
                       if os.path.isfile(os.path.join(folder_path, f)) and
                       os.path.splitext(f)[1].lower() in image_extensions]

        if not image_files:
            print("在指定文件夹中未找到图片文件！")
            return

        # 按文件名排序[1,2](@ref)
        image_files.sort()

        print(f"找到 {len(image_files)} 张图片，开始重命名...")

        # 遍历并重命名图片文件[1,5](@ref)
        for index, filename in enumerate(image_files, start=start_index):
            # 获取文件扩展名[1](@ref)
            file_extension = os.path.splitext(filename)[1]

            # 构建新文件名[4,5](@ref)
            new_filename = f"{prefix}{index:0{digits}d}{file_extension}"

            # 构建完整路径[1](@ref)
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # 检查新文件名是否已存在[7](@ref)
            if os.path.exists(new_path):
                print(f"警告：{new_filename} 已存在，跳过重命名 {filename}")
                continue

            # 重命名文件[1,7](@ref)
            os.rename(old_path, new_path)
            print(f"重命名成功：{filename} -> {new_filename}")

        print("所有图片重命名完成！")

    except FileNotFoundError:
        print(f"错误：文件夹 {folder_path} 不存在")
    except PermissionError:
        print("错误：没有访问该文件夹的权限")
    except Exception as e:
        print(f"发生错误：{e}")


# 使用方法
if __name__ == "__main__":
    # 设置你的图片文件夹路径[1](@ref)
    folder_path = r"F:\BaiduNetdiskDownload\板栗\archive\chestnut_improve\VOC2007\JPEGImages"  # 请修改为你的实际路径

    # 调用函数进行重命名
    batch_rename_images(folder_path, start_index=1, prefix="IMG_", digits=3)