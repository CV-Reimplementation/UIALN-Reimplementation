import os

import cv2
import kornia.color
import numpy as np
import torch
from einops import rearrange
from kornia.color import LabToRgb
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from DataLoader import Test_data
from Net import Retinex_Decomposition_net, Illumination_Correction, AL_Area_Selfguidance_Color_Correction, \
    Detail_Enhancement, Channels_Fusion


def to_LAB_range(LAB):
    L = LAB[:, 0, :, :]
    AB = LAB[:, 1:, :, :]
    L = L * 100
    AB = AB * 256 - 128
    L = L.unsqueeze(0)

    return torch.cat((L, AB), dim=1)

def test(input_dir, output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("模型导入")
    model_1 = Retinex_Decomposition_net().to(device)
    model1_path = './save_model/Retinex_Light_Correction_net.pth'
    model_1.load_state_dict(torch.load(model1_path)['model'])

    model_2 = Illumination_Correction().to(device)
    model2_path = './save_model/Illumination_Correction_net.pth'
    model_2.load_state_dict(torch.load(model2_path)['model'])

    model_3 = AL_Area_Selfguidance_Color_Correction().to(device)
    model3_path = './save_model/AL_Area_Selfguidance_Color_Correction_net.pth'
    model_3.load_state_dict(torch.load(model3_path)['model'])

    model_4 = Detail_Enhancement().to(device)
    model4_path = './save_model/Detail_Enhancement_net.pth'
    model_4.load_state_dict(torch.load(model4_path)['model'])

    model_fusion = Channels_Fusion().to(device)
    model_fusion_path = './save_model/Channels_Fusion_net.pth'
    model_fusion.load_state_dict(torch.load(model_fusion_path)['model'])
    print("模型导入完成")

    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_fusion.eval()

    dataset = Test_data(input_dir, size=256)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    rgb = LabToRgb()
    for _, batch in enumerate(tqdm(train_loader)):
        ABcc = batch[0].to(device)
        L = batch[1].to(device)

        with torch.no_grad():
            temp = model_1(L)
            I_light, R_light = torch.split(temp, 1, dim=1)
            I_delight = model_2(temp)
            M_image = I_light - I_delight
            ABcc = model_3(M_image, ABcc)
            L_delight = I_delight * R_light

            L_en = model_4(L_delight)  # enhanced L
            LAB = torch.cat((L_en, ABcc), dim=1)
            LAB = model_fusion(LAB)
            LAB = to_LAB_range(LAB)

            img_rgb = rgb(LAB)

        save_image(img_rgb, os.path.join(output_dir, batch[2][0]))


def test_all():
    input_root = '/data/underwater/UIEB-EUVP-LSUI2/'
    input_folders = ['test-UIEB', 'test-EUVP', 'test-LSUI',
                     'UIEB-challenging-60', 'EUVP-Unpaired-test', 'test-RUIE-nonref', 'test-seathru-nonref']
    output_folders = ['UIEB', 'EUVP', 'LSUI', 'UIEB', 'EUVP', 'RUIE', 'Seathru']

    output_root = './result/UIALN/'

    for i, (input_folder, output_folder) in enumerate(zip(input_folders, output_folders)):
        input_dir = os.path.join(input_root, input_folder, 'input')
        if i < 3:
            output_dir = os.path.join(output_root, 'Ref', output_folder)
        else:
            output_dir = os.path.join(output_root, 'Non', output_folder)
        test(input_dir, output_dir)

if __name__ == '__main__':
    test_all()
