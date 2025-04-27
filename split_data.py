import random
import glob
import datetime
import os


def split_data(dir_root):
    random.seed(0)
    ori_img = glob.glob(dir_root + r"\*b.png")
    filenames = [os.path.basename(file) for file in ori_img]
    ori_img2 = []
    for i in range(0, len(ori_img)):
        filename = filenames[i]
        time = datetime.datetime(year=int(filename[7:11]), month=int(filename[11:13]), day=int(filename[13:15]),
                                 hour=int(filename[16:18]), minute=int(filename[18:20]),
                                 second=0)
        ori_img2.append(str(time)[:10])
    ori_img2 = sorted(set((ori_img2)))
    k = 0.1
    sample_data = sorted(random.sample(population=ori_img2, k=int(k * len(ori_img2))))
    sample_data2 = []
    for i in range(0, len(ori_img)):
        filename = filenames[i]
        time = datetime.datetime(year=int(filename[7:11]), month=int(filename[11:13]), day=int(filename[13:15]),
                                 hour=int(filename[16:18]), minute=int(filename[18:20]),
                                 second=0)
        if str(time)[:10] in sample_data:
            sample_data2.append(ori_img[i])
    train_ori_imglist = []
    val_ori_imglist = []
    for img in ori_img:
        if img in sample_data2:
            val_ori_imglist.append(img)
        else:
            train_ori_imglist.append(img)
    return train_ori_imglist, val_ori_imglist


if __name__ == '__main__':
    a, b = split_data(r"D:\merge")