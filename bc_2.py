import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
# import pandas as pd
import numpy as np
import time

class FlightDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        folders = glob.glob(os.path.join(root_dir, 'run_*'))
        for folder in folders:
            state_file = os.path.join(folder, 'current_flight_pos.txt')
            control_file = os.path.join(folder, 'flight_control_list.txt')
            missile_file = os.path.join(folder, 'missile_sim.txt')
            
            states = np.loadtxt(state_file, dtype=np.float32)
            # print("飞机状态量加载完毕")
            controls = np.loadtxt(control_file, dtype=np.float32)
            # print("飞机控制量加载完毕")
            missiles = np.loadtxt(missile_file, dtype=np.float32)
            # print("导弹状态量加载完毕")
            
            if len(states) == len(controls) == len(missiles):
                for state, missile, control in zip(states, missiles, controls):
                    combined_state = np.concatenate((state, missile))
                    self.data.append((combined_state, control))
        self.data = np.array(self.data, dtype=object)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, control = self.data[idx]
        return state, control

# 简单的前馈神经网络模型
class BehaviorCloningModel(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(BehaviorCloningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs, device, writer, checkpoint_path):
    model.to(device)
    start_time = time.time()#方便查询时间
    
    # Check if a checkpoint exists and load it
    start_epoch = 0
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for states, controls in dataloader:
            states, controls = states.to(device), controls.to(device)
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, controls)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪,防止爆炸
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        elapsed_time = time.time() - epoch_start_time
        total_elapsed_time = time.time() - start_time
        estimated_total_time = total_elapsed_time / (epoch + 1) * num_epochs
        remaining_time = estimated_total_time - total_elapsed_time

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
              f'Elapsed time: {elapsed_time:.2f}s, '
              f'Total time: {total_elapsed_time:.2f}s, '
              f'Estimated remaining time: {remaining_time:.2f}s')
        
        writer.add_scalar('Loss/train', avg_loss, epoch)

def test_model(model, dataloader, criterion, device, writer, epoch):
    model.to(device)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for states, controls in dataloader:
            states, controls = states.to(device), controls.to(device)
            outputs = model(states)
            loss = criterion(outputs, controls)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(dataloader)
    print(f'Test Loss: {avg_test_loss:.4f}')
    writer.add_scalar('Loss/test', avg_test_loss, epoch)
    return avg_test_loss

def predict_control(model, state, device):
    state = torch.tensor(state, dtype=torch.float32).to(device)
    state = state.unsqueeze(0)  # 添加batch维度
    model.eval()
    with torch.no_grad():
        control = model(state)
    control = control.cpu().numpy().squeeze()
    return control

# 参数
root_dir = './hit_add'  # 为数据文件夹路径
test_root_dir = './hit_true'  # 为测试数据文件夹路径
# csv_file = 'trajectory_data.csv'  # 为数据文件
state_dim = 6 # 飞机状态量的维度
missile_dim = 6  # 导弹状态量的维度
input_dim = state_dim + missile_dim  # 总的输入维度
action_dim = 3  # 控制量的维度，nx、ny、γ
batch_size = 128
num_epochs = 150
learning_rate = 0.001
checkpoint_path = os.path.join(root_dir, 'checkpoint.pth')
model_path = './hit_add/behavior_cloning_model_batch128_epoach150_lr0.001.pth'

# 检查GPU是否可用，并设置为显卡0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# TensorBoard writer
writer = SummaryWriter(log_dir='runs/behavior_cloning_experiment') 

# 数据集和数据加载器
train_dataset  = FlightDataset(root_dir)
train_dataloader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = FlightDataset(test_root_dir)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # 模型、损失函数和优化器
# model = BehaviorCloningModel(input_dim, action_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def main(mode='train', state_input=None):
    # Initialize the criterion
    criterion = nn.MSELoss()

    if mode == 'train':# 训练模型
        model = BehaviorCloningModel(input_dim, action_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_model(model, train_dataloader, criterion, optimizer, num_epochs, device, writer, checkpoint_path)
        final_model_path  = os.path.join(root_dir, 'BC_model_final_batch128_epoach150_lr0.001.pth')
        torch.save(model.state_dict(), final_model_path )
        print(f'Model saved to {final_model_path }')

    elif mode == 'test':# 测试模型
        model = BehaviorCloningModel(input_dim, action_dim).to(device)
        model.load_state_dict(torch.load(model_path))
        print(f'加载模型完毕')
        test_loss = test_model(model, test_dataloader, criterion, device, writer, num_epochs)
        print(f'Test Loss: {test_loss}')
    elif mode == 'predict':
        if state_input is None:
            raise ValueError("For prediction mode, state_input must be provided")
        model = BehaviorCloningModel(input_dim, action_dim)
        model.load_state_dict(torch.load(model_path))
        print(f'加载模型完毕')
        control_output = predict_control(model, state_input, device)
        print(f'Input state: {state_input}')
        print(f'Predicted control: {control_output}')
        return control_output

# 关闭 TensorBoard writer
# writer.close()

# 示例调用
if __name__ == '__main__':
    mode = 'predict'  # 可以选择 'train', 'test', 或 'predict'
    if mode == 'predict':
        state_example = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32)
        main(mode=mode, state_input=state_example)
    else:
        main(mode=mode)


