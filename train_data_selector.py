import random
import shutil
import os

ori_dir_al = '/data/underwater/UIALN/Synthetic_dataset/dataset_with_AL/train_bak'
ori_dir_no_al = '/data/underwater/UIALN/Synthetic_dataset/dataset_no_AL_bak'

dest_dir_al = '/data/underwater/UIALN/Synthetic_dataset/dataset_with_AL/train'
dest_dir_no_al = '/data/underwater/UIALN/Synthetic_dataset/dataset_no_AL'

folder_list = os.listdir(ori_dir_al)

for folder in folder_list:
    folder_dir = os.path.join(ori_dir_al, folder)
    img_list = os.listdir(folder_dir)
    random.shuffle(img_list)
    img_list = img_list[0:200]
    for img in img_list:
        ori_img_al_path = os.path.join(folder_dir, img)
        ori_img_no_al_path = os.path.join(ori_dir_no_al, folder, img)

        dest_img_al_path = os.path.join(dest_dir_al, img)
        dest_img_no_al_path = os.path.join(dest_dir_no_al, img)

        shutil.copy(ori_img_al_path, dest_img_al_path)
        shutil.copy(ori_img_no_al_path, dest_img_no_al_path)

