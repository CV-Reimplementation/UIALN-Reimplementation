from tqdm import tqdm

from Net import *
from Loss import *
from DataLoader import *
from torch.utils.data import DataLoader
import time

consLoss = nn.MSELoss()
recLoss = nn.MSELoss()
colorLoss = nn.MSELoss()
hazeLoss = nn.MSELoss()
# structure-aware TV loss
smoothLoss = TVLoss()

def train_1(device, start_epoch, train_batch_size, learning_rate, num_epochs, save_point, L_no_light_path, L_light_path):
    print("模型导入中")
    model = Retinex_Decomposition_net().to(device)
    if start_epoch != 0:
        model_path = './checkpoints/Retinex_Decomposition_net/epoch_' + str(start_epoch) + '.pth'
        model.load_state_dict(torch.load(model_path)['model'])
    print("模型导入完成")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = retinex_decomposition_data(L_no_light_path, L_light_path)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    total_loss = 0
    for epoch in range(start_epoch+1, num_epochs+1):
        print("epoch: ", epoch)
        start_time = time.time()

        i = 0
        epoch_loss = 0
        for _, batch in enumerate(tqdm(train_loader)):
            i += 1
            L_no_light = batch[0].to(device)
            L_light = batch[1].to(device)
            L_no_light_hat = model(L_no_light)
            # 每个batch中的第一个是I_no_light_hat，第二个是R_no_light_hat，它们的shape都是[batch_size, 1, 256, 256]，batch不改变
            I_no_light_hat, R_no_light_hat = torch.split(L_no_light_hat, 1, dim=1)
            L_light_hat = model(L_light)
            I_light_hat, R_light_hat = torch.split(L_light_hat, 1, dim=1)
            loss_1 = consLoss(R_light_hat, R_no_light_hat)
            loss_2_1 = recLoss(I_light_hat*R_light_hat, L_light)
            loss_2_2 = recLoss(I_no_light_hat*R_no_light_hat, L_no_light)
            loss_3 = smoothLoss(I_light_hat, R_light_hat)
            loss_4 = smoothLoss(I_no_light_hat, R_no_light_hat)
            loss = loss_1 + loss_2_1 + loss_2_2 + loss_3 + loss_4
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % save_point == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, './checkpoints/Retinex_Decomposition_net/epoch_' + str(epoch) + '.pth')
            torch.save(state, './save_model/Retinex_Light_Correction_net.pth')
        time_epoch = time.time() - start_time
        epoch_loss = epoch_loss*1.0/i
        total_loss += epoch_loss
        print("train_1==>No: {} epoch, time: {:.2f}, loss: {:.5f}".format(epoch, time_epoch / 60, epoch_loss))
        with open("output.txt", "a") as f:
            f.write("train_1==>No: {} epoch, time: {:.2f}, loss: {:.5f}\n".format(epoch, time_epoch / 60, epoch_loss))

    print("total_loss:", total_loss*1.0/num_epochs-start_epoch)


def train_2(device, start_epoch, train_batch_size, learning_rate, num_epochs, save_point, L_no_light_path, L_light_path):
    print("模型导入")
    # 前置模型
    model_1 = Retinex_Decomposition_net().to(device)
    model1_path = './save_model/Retinex_Light_Correction_net.pth'
    model_1.load_state_dict(torch.load(model1_path)['model'])
    # 后置模型
    model_2 = Illumination_Correction().to(device)
    if start_epoch != 0:
        model2_path = './checkpoints/Illumination_Correction/epoch_' + str(start_epoch) + '.pth'
        model_2.load_state_dict(torch.load(model2_path)['model'])
    print("模型导入完成")
    model_1.eval()
    model_2.train()
    optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate)

    dataset = retinex_decomposition_data(L_no_light_path, L_light_path)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    total_loss = 0
    for epoch in range(start_epoch+1, num_epochs+1):
        print("epoch: ", epoch)
        start_time = time.time()
        i = 0
        epoch_loss = 0
        for _, batch in enumerate(tqdm(train_loader)):
            i+=1
            L_no_light = batch[0].to(device)
            L_light = batch[1].to(device)
            temp = model_1(L_light)
            I_light, R_light = torch.split(temp, 1, dim=1)
            temp = model_1(L_no_light)
            I_no_light, R_no_light = torch.split(temp, 1, dim=1)
            I_delight_hat = model_2(torch.cat((I_light, R_light), dim=1))
            # 感觉论文这里有点问题，之后问一下
            loss_1 = recLoss(I_delight_hat*R_light, L_no_light)
            loss_2 = consLoss(R_light, R_no_light)
            loss = loss_1 + loss_2
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % save_point == 0:
            state = {'model': model_2.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, './checkpoints/Illumination_Correction/epoch_' + str(epoch) + '.pth')
            torch.save(state, './save_model/Illumination_Correction_net.pth')
        time_epoch = time.time() - start_time
        epoch_loss = epoch_loss*1.0/i
        total_loss += epoch_loss
        print("train_2==>No: {} epoch, time: {:.2f}, loss: {:.5f}".format(epoch, time_epoch / 60, epoch_loss))
        with open("output.txt", "a") as f:
            f.write("train_2==>No: {} epoch, time: {:.2f}, loss: {:.5f}\n".format(epoch, time_epoch / 60, epoch_loss))
    print("total_loss:",total_loss*1.0/num_epochs-start_epoch)


def train_3(device, start_epoch, train_batch_size, learning_rate, num_epochs, save_point, ABcc_path, gt_path, size):
    print("模型导入")
    # 前置双模型
    model_1 = Retinex_Decomposition_net().to(device)
    model1_path = './save_model/Retinex_Light_Correction_net.pth'
    model_1.load_state_dict(torch.load(model1_path)['model'])
    model_2 = Illumination_Correction().to(device)
    model2_path = './save_model/Illumination_Correction_net.pth'
    model_2.load_state_dict(torch.load(model2_path)['model'])
    # 后置模型
    model_3 = AL_Area_Selfguidance_Color_Correction().to(device)
    if start_epoch != 0:
        model3_path = './checkpoints/AL_Area_Selfguidance_Color_Correction/epoch_' + str(start_epoch) + '.pth'
        model_3.load_state_dict(torch.load(model3_path)['model'])
    print("模型导入完成")
    model_1.eval()
    model_2.eval()
    model_3.train()
    optimizer = torch.optim.Adam(model_3.parameters(), lr=learning_rate)

    dataset = AL_data(ABcc_path, gt_path, size=size)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    total_loss = 0
    for epoch in range(start_epoch+1, num_epochs+1):
        print("epoch: ", epoch)

        start_time = time.time()
        i = 0
        epoch_loss = 0
        for _, batch in enumerate(tqdm(train_loader)):
            i+=1
            ABcc = batch[0].to(device)
            gt = batch[1].to(device)
            L = batch[2].to(device)
            temp = model_1(L)
            I_light, R_light = torch.split(temp, 1, dim=1)
            I_delight = model_2(temp)
            M_image = I_light - I_delight
            ABcc_hat = model_3(M_image, ABcc)
            loss = colorLoss(ABcc_hat, gt)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % save_point == 0:
            state = {'model': model_3.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, './checkpoints/AL_Area_Selfguidance_Color_Correction/epoch_' + str(epoch) + '.pth')
            torch.save(state, './save_model/AL_Area_Selfguidance_Color_Correction_net.pth')
        time_epoch = time.time() - start_time
        epoch_loss = epoch_loss*1.0/i
        total_loss += epoch_loss
        print("train_3==>No: {} epoch, time: {:.2f}, loss: {:.5f}".format(epoch, time_epoch / 60, epoch_loss))
        with open("output.txt", "a") as f:
            f.write("train_3==>No: {} epoch, time: {:.2f}, loss: {:.5f}\n".format(epoch, time_epoch / 60, epoch_loss))
    print("total_loss:",total_loss*1.0/num_epochs-start_epoch)


def train_4(device, start_epoch, train_batch_size, learning_rate, num_epochs, save_point, ABcc_path, gt_path, size):
    print("模型导入")
    # 前置模型
    model_1 = Retinex_Decomposition_net().to(device)
    model1_path = './save_model/Retinex_Light_Correction_net.pth'
    model_1.load_state_dict(torch.load(model1_path)['model'])
    model_2 = Illumination_Correction().to(device)
    model2_path = './save_model/Illumination_Correction_net.pth'
    model_2.load_state_dict(torch.load(model2_path)['model'])
    model_3 = AL_Area_Selfguidance_Color_Correction().to(device)
    model3_path = './save_model/AL_Area_Selfguidance_Color_Correction_net.pth'
    model_3.load_state_dict(torch.load(model3_path)['model'])

    # 后置模型
    model_4 = Detail_Enhancement().to(device)
    model_fusion = Channels_Fusion().to(device)
    if start_epoch != 0:
        model4_path = './checkpoints/Detail_Enhancement/epoch_' + str(start_epoch) + '.pth'
        model_4.load_state_dict(torch.load(model4_path)['model'])
        model_fusion_path = './checkpoints/Channels_Fusion/epoch_' + str(start_epoch) + '.pth'
        model_fusion.load_state_dict(torch.load(model_fusion_path)['model'])
    print("模型导入完成")
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.train()
    model_fusion.train()

    optimizer_4 = torch.optim.Adam(model_4.parameters(), lr=learning_rate)
    optimizer_fusion = torch.optim.Adam(model_fusion.parameters(), lr=learning_rate)
    dataset = Detail_Enhancement_data(ABcc_path, gt_path, size=size)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    total_loss = 0
    for epoch in range(start_epoch + 1, num_epochs + 1):
        print("epoch: ", epoch)

        start_time = time.time()
        i = 0
        epoch_loss = 0
        for _, batch in enumerate(tqdm(train_loader)):
            i += 1
            ABcc = batch[0].to(device)
            L = batch[1].to(device)
            gt_L_tensor = batch[2].to(device)
            gt = batch[3].to(device)

            temp = model_1(L)
            I_light, R_light = torch.split(temp, 1, dim=1)
            I_delight = model_2(temp)
            M_image = I_light - I_delight
            ABcc = model_3(M_image, ABcc)
            L_delight = I_delight * R_light

            L_en_hat = model_4(L_delight)  # enhanced L
            LAB_hat = torch.cat((L_en_hat, ABcc), dim=1)
            LAB_hat = model_fusion(LAB_hat)

            loss_haze = hazeLoss(gt_L_tensor, L_en_hat)
            loss_recons = recLoss(gt, LAB_hat)
            final_loss = loss_haze + loss_recons
            epoch_loss += final_loss

            optimizer_fusion.zero_grad()
            optimizer_4.zero_grad()
            final_loss.backward()
            optimizer_fusion.step()

            # final_loss.backward()
            optimizer_4.step()
        if epoch % save_point == 0:
            state = {'model': model_4.state_dict(), 'optimizer': optimizer_4.state_dict(), 'epoch': epoch}
            torch.save(state, './checkpoints/Detail_Enhancement/epoch_' + str(epoch) + '.pth')
            torch.save(state, './save_model/Detail_Enhancement_net.pth')
            state = {'model': model_fusion.state_dict(), 'optimizer': optimizer_fusion.state_dict(), 'epoch': epoch}
            torch.save(state, './checkpoints/Channels_Fusion/epoch_' + str(epoch) + '.pth')
            torch.save(state, './save_model/Channels_Fusion_net.pth')

        time_epoch = time.time() - start_time
        epoch_loss = epoch_loss * 1.0 / i
        total_loss += epoch_loss
        print("train_4==>No: {} epoch, time: {:.2f}, loss: {:.5f}".format(epoch, time_epoch / 60, epoch_loss))
        with open("output.txt", "a") as f:
            f.write("train_4==>No: {} epoch, time: {:.2f}, loss: {:.5f}\n".format(epoch, time_epoch / 60, epoch_loss))
    print("total_loss:", total_loss * 1.0 / num_epochs - start_epoch)


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_epoch = 0
    train_batch_size = 64
    learning_rate = 0.0002
    num_epochs = 200
    save_point = 20
    L_no_light_path = "./dataset/UIALN_dataset/Synthetic_dataset/dataset_no_AL"
    L_light_path = "./dataset/UIALN_dataset/Synthetic_dataset/dataset_with_AL/train"
    train_1(device, start_epoch, train_batch_size, learning_rate, num_epochs, save_point, L_no_light_path, L_light_path)

    start_epoch = 0
    train_batch_size = 48
    learning_rate = 0.0002
    num_epochs = 200
    save_point = 20
    L_no_light_path = "./dataset/UIALN_dataset/Synthetic_dataset/dataset_no_AL"
    L_light_path = "./dataset/UIALN_dataset/Synthetic_dataset/dataset_with_AL/train"
    train_2(device, start_epoch, train_batch_size, learning_rate, num_epochs, save_point, L_no_light_path, L_light_path)

    start_epoch = 0
    train_batch_size = 24
    learning_rate = 0.0002
    num_epochs = 200
    save_point = 20
    ABcc_path = "/data/underwater/UIEB-EUVP-LSUI2/train/input"
    gt_path = "/data/underwater/UIEB-EUVP-LSUI2/train/target"
    size = 256
    train_3(device, start_epoch, train_batch_size, learning_rate, num_epochs, save_point, ABcc_path, gt_path, size)

    start_epoch = 0
    train_batch_size = 6
    learning_rate = 0.0002
    num_epochs = 200
    save_point = 20
    ABcc_path = "/data/underwater/UIEB-EUVP-LSUI2/train/input"
    gt_path = "/data/underwater/UIEB-EUVP-LSUI2/train/target"
    size = 256
    train_4(device, start_epoch, train_batch_size, learning_rate, num_epochs, save_point, ABcc_path, gt_path, size)


if __name__ == '__main__':
    train()

