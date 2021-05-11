import os


def files_and_dirs_list(dir_path):
    """
    遍历文件夹及文件夹下所有文件（包括文件夹）
    :param dir_path: 文件夹路径
    :return:
    	root 所指的是当前正在遍历的这个文件夹的本身的地址
    	dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    	files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    """
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file == ".DS_Store":
                print("file_path:", os.path.join(root, file))
                os.remove(os.path.join(root, file))


if __name__ == "__main__":
    # 获取当前文件的目录
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.join("/home/felixfu/data/segmentation/")
    files_and_dirs_list(root_path)