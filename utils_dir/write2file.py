import numpy as np

def write2file(file_dir, ternsor_file):
    tensor2numpy = ternsor_file.cpu().numpy()
    with open(file_dir, 'a') as f:
        for i, x in enumerate(tensor2numpy):
            np.savetxt(f, [x])
            if i == len(tensor2numpy) - 1:
               f.write('\n')

def list2file(file_dir, list_data):
    with open(file_dir, 'a') as f:
        for item in list_data:
           item = item.cpu().numpy()
           f.write(str(item))
           f.write(' ')
        f.write('\n')

def write_cosine2file(file_dir, ternsor_file):
    """
       file_dir: 保存文件的路径
       ternsor_fil: 需要保存的数据
       description: 保存文件头描述
    """
    tensor2numpy = ternsor_file.cpu().numpy()
    with open(file_dir, 'a') as f:
        for i, x in enumerate(tensor2numpy):
            f.write(str(x[6]) + ' ')

def write_cosine3file(file_dir, ternsor_file):
    """
       file_dir: 保存文件的路径
       ternsor_fil: 需要保存的数据
       description: 保存文件头描述
    """
    tensor2numpy = ternsor_file.cpu().numpy()
    with open(file_dir, 'a') as f:
        for i, x in enumerate(tensor2numpy):
            f.write(str(x[4]) + ' ')


def write_cosine2file1(file_dir, ternsor_file):
    """
       file_dir: 保存文件的路径
       ternsor_fil: 需要保存的数据
       description: 保存文件头描述
    """
    tensor2numpy = ternsor_file.cpu().numpy()
    with open(file_dir, 'a') as f:
        f.write(str(tensor2numpy) + ' ')




   