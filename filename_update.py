import os


def convert_jpg_extension(folder_path):
    """
    将指定文件夹中所有.JPG文件扩展名改为.jpg

    参数:
    folder_path: 要处理的文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return

    # 计数器
    count = 0

    print(f"开始处理文件夹: {folder_path}")

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否以.JPG结尾（不区分大小写）
        if filename.lower().endswith('.jpg'):
            # 获取文件完整路径
            old_path = os.path.join(folder_path, filename)

            # 如果扩展名是大写的.JPG
            if filename.endswith('.JPG'):
                # 创建新文件名（小写扩展名）
                new_filename = filename[:-4] + '.jpg'
                new_path = os.path.join(folder_path, new_filename)

                # 重命名文件
                os.rename(old_path, new_path)
                print(f"已重命名: {filename} -> {new_filename}")
                count += 1
            # 如果已经是小写.jpg，不需要修改
            else:
                print(f"跳过（已小写）: {filename}")

    print(f"\n处理完成！共修改了 {count} 个文件")


if __name__ == "__main__":
    # 设置要处理的文件夹路径
    folder_path = input("请输入要处理的文件夹路径: ").strip()

    # 调用函数处理
    convert_jpg_extension(folder_path)