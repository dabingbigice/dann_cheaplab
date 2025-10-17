import os

def extract_image_names_to_txt(folder_path, output_file="image_names.txt"):
    """
    提取文件夹中所有图片文件名（不含后缀）并保存到TXT文件

    Parameters:
    folder_path (str): 图片文件夹路径
    output_file (str): 输出的TXT文件名，默认为image_names.txt
    """
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"错误：文件夹 '{folder_path}' 不存在")
            return False

        # 定义常见的图片扩展名
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

        # 获取文件夹中所有文件
        all_files = os.listdir(folder_path)

        # 过滤出图片文件并去除后缀
        image_files = []
        for f in all_files:
            if f.lower().endswith(image_extensions):
                # 使用os.path.splitext去除文件后缀[1,2](@ref)
                name_without_extension = os.path.splitext(f)[0]
                image_files.append(name_without_extension)

        # 按文件名排序
        image_files.sort()

        # 将图片文件名（不含后缀）写入TXT文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for filename in image_files:
                f.write(filename + '\n')

        print(f"成功提取 {len(image_files)} 个图片文件名（不含后缀）到 {output_file}")
        return True

    except Exception as e:
        print(f"发生错误：{e}")
        return False

if __name__ == "__main__":
    # 获取用户输入的文件夹路径
    folder_path = input("请输入图片文件夹路径: ").strip()

    # 可选：自定义输出文件名
    output_name = input("请输入输出文件名（直接回车使用默认值image_names.txt）: ").strip()
    output_file = output_name if output_name else "image_names.txt"

    # 执行提取操作
    extract_image_names_to_txt(folder_path, output_file)