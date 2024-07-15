import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time

# 参数
root_dir = r'/home/zjm/Workspace/imitation_learning/hit_false'# 为数据文件夹路径
test_root_dir = './hit_true'  # 为测试数据文件夹路径
checkpoint_root_path= r"./checkpoints/"
state_dim = 6 # 飞机状态量的维度
missile_dim = 6  # 导弹状态量的维度
input_dim = state_dim + missile_dim  # 总的输入维度
sequence_length = 100
total_input_dim = input_dim * sequence_length
action_dim = 3  # 控制量的维度，nx、ny、γ
output_dim = sequence_length * action_dim
batch_size = 512
num_epochs = 150
learning_rate = 0.001
checkpoint_path = os.path.join(checkpoint_root_path, 'checkpoint.pth')
model_path = './result/BC_model_Transformer_batch512_epoach150_001.pth'

# 检查GPU是否可用，并默认设置为显卡0
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
# 如果目录不存在，则创建目录
os.makedirs(os.path.dirname(model_path), exist_ok=True)

class FlightDataset(Dataset):
    def __init__(self, root_dir, sample_size=100):
        self.data = []
        self.sample_size = sample_size
        folders = glob.glob(os.path.join(root_dir, 'run_*'))
        for folder in folders:
            state_file = os.path.join(folder, 'current_flight_pos.txt')
            control_file = os.path.join(folder, 'flight_control_list.txt')
            missile_file = os.path.join(folder, 'missile_sim.txt')

            states = np.loadtxt(state_file, dtype=np.float32)
            controls = np.loadtxt(control_file, dtype=np.float32)
            missiles = np.loadtxt(missile_file, dtype=np.float32)

            min_length = min(len(states), len(missiles))
            states = states[:min_length]
            missiles = missiles[:min_length]

            for i in range(0, min_length, self.sample_size):
                combined_state = np.concatenate((states[i:i + self.sample_size], missiles[i:i + self.sample_size]),
                                                axis=1)

                if combined_state.shape[0] < self.sample_size:
                    pad_length = self.sample_size - combined_state.shape[0]
                    combined_state = np.pad(combined_state, ((0, pad_length), (0, 0)), 'constant', constant_values=0)

                control_idx = min(i // self.sample_size, len(controls) - self.sample_size)
                control = controls[control_idx:control_idx + self.sample_size].flatten()
                self.data.append((combined_state, control))

        self.data = np.array(self.data, dtype=object)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, control = self.data[idx]
        return state, control

    def print_total_rows(self):
        print(f'Total number of rows: {len(self.data) * self.sample_size}')


class TransformerModel(nn.Module):
    def __init__(self, input_dim, action_dim, nhead=4, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, dim_feedforward))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(dim_feedforward, action_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)  # 将输入展平以适应嵌入层
        x = self.embedding(x)  # 应用嵌入层
        x = x.view(batch_size, seq_len, -1)  # 重塑回 (batch_size, seq_len, dim_feedforward)
        x = x + self.positional_encoding[:, :seq_len, :]  # 加上位置编码
        x = self.transformer_encoder(x)  # 应用Transformer编码器
        x = self.fc_out(x).view(batch_size, seq_len, 3)  # 最终输出重塑
        return x


def train_model(model, dataloader, criterion, optimizer, num_epochs, device, checkpoint_pat,writer):
    model.to(device)
    start_time = time.time()#方便查询时间

    # Check if a checkpoint exists and load it
    start_epoch = 0
    best_loss = float('inf')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', best_loss)
        print(f"从第 {start_epoch} 个epoch恢复训练")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for states, controls in dataloader:
            states, controls = states.to(device), controls.to(device)

            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, controls.view(-1, 100, 3))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        elapsed_time = time.time() - epoch_start_time
        total_elapsed_time = time.time() - start_time
        estimated_total_time = total_elapsed_time / (epoch + 1) * num_epochs
        remaining_time = estimated_total_time - total_elapsed_time

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, '
              f'Elapsed time: {elapsed_time:.2f}s, '
              f'Total time: {total_elapsed_time:.2f}s, '
              f'Estimated remaining time: {remaining_time:.2f}s')
        
        # 使用TensorBoard记录损失
        writer.add_scalar('Training Loss', avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'保存新的最佳模型检查点，损失值: {best_loss:.4f}')
    # 关闭 TensorBoard writer
    writer.close()


def main(mode='train', state_input=None):
    # Initialize the criterion
    criterion = nn.MSELoss()
    # TensorBoard writer
    writer = SummaryWriter(log_dir='runs/behavior_cloning_experiment') 

    if mode == 'train':
        train_dataset = FlightDataset(root_dir)
        train_dataset.print_total_rows()

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = TransformerModel(input_dim, action_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_model(model, train_dataloader, criterion, optimizer, num_epochs, device, checkpoint_path, writer)
        torch.save(model.state_dict(), model_path)
        print('训练完成，模型已保存。')
    else:
        model = TransformerModel(input_dim, action_dim).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        state_input = torch.tensor(state_input, dtype=torch.float32).to(device)
        with torch.no_grad():
            control_output = model(state_input)
        return control_output


if __name__ == "__main__":
    main()
