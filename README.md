# Behavior Cloning Project

This project is for reinforce learning in imitation learing area. We use ourselves dataset to genertate the original agent. All the project is responsible to SJTU, China ShangHai BATC lab.

This repository contains the implementation of behavior cloning for agent training using different network architectures and monitoring tools. 

该项目实现了导弹和目标的比例导航与机动库结合的模拟。项目包括使用 Transformer 模型进行预测控制，利用已有模型进行模拟，以及模型的训练过程。

The project includes three main scripts:
## Old Files
1. **train_bc.py**: The first version of behavior cloning code.
2. **train_bc_2.py**: The second version with TensorBoard monitoring.
3. **train_bc_trans.py**: Utilizing Transformer as the backbone.

## New Files
4. **models.py**：定义了 `TransformerModel` 类，用于创建 Transformer 模型。
5. **utils.py**：包含各种工具函数，如预测控制、数据加载和处理、Runge-Kutta 积分等。
6. **missile.py**：定义了 `Missile` 类，包含导弹的属性和方法。
7. **flight.py**：定义了 `Flight` 类，包含飞行器的属性和方法。
8. **main.py**：主程序，负责加载模型、准备数据并进行预测和模拟。
9. **train_xxx.py**：包含模型训练的代码，负责加载数据、训练模型并保存训练好的模型，这里代指老文件123。

## Project Structure

```
├── bc.py
├── bc_2.py
├── bc_trans.py
├── models.py               # Transformer 模型定义
├── utils.py                # 工具函数，包括数据加载、预处理等
├── missile.py              # 导弹类定义
├── flight.py               # 飞行器类定义
├── main.py                 # 主程序，进行模型预测和模拟
├── train.py                # 训练脚本，进行模型训练
├── README.md
└── data
    └── hit_false
```
## 依赖项

项目所需的主要依赖项如下：

- numpy
- scipy
- matplotlib
- torch
- pandas

## Scripts Overview

### 1. `bc.py`
This is the first version of the behavior cloning code. The backbone is a three-layer fully connected network with batch normalization (BN) in the intermediate layers. This version does not include TensorBoard logging for training and testing.

#### Key Features:
- Three-layer fully connected network.
- Batch normalization in intermediate layers.
- No TensorBoard logging.

### 2. `bc_2.py`
The second version of the behavior cloning code. It uses the same network structure as `bc.py` but includes TensorBoard for monitoring the loss during training and testing.

#### Key Features:
- Three-layer fully connected network.
- Batch normalization in intermediate layers.
- TensorBoard logging for monitoring loss.

### 3. `bc_trans.py`
This script uses a Transformer as the backbone for behavior cloning.

#### Key Features:
- Transformer backbone for behavior cloning.

## Dataset

The best dataset so far is `hit_false`, which contains scenarios where the agent was not hit by the missiles. This dataset has proven to be effective for training the agent to avoid missiles.

## Training Results

Using the `hit_false` dataset, the trained agent has achieved the ability to avoid missiles successfully but does not hit any targets.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hollyming/imitation-learning-plane.git
   cd behavior-cloning-project
   ```

2. **Install dependencies:**
   Ensure you have the necessary libraries installed. You can use `requirements.txt` if provided, or install manually.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the scripts of training:**

在开始模型训练之前，请确保数据文件 `flight_pos_file` 和 `missile_sim_file` 存在并路径正确。可以在 `train.py` 中设置相应路径。

运行训练脚本：

```bash
python train.py
```

   - To run the first version (`bc.py`):
     ```bash
     python bc.py
     ```
   - To run the second version (`bc_2.py`):
     ```bash
     python bc_2.py
     ```
   - To run the Transformer version (`bc_trans.py`):
     ```bash
     python bc_trans.py
     ```

4. **Monitor Training (For `bc_2.py`):**
   - Start TensorBoard:
     ```bash
     tensorboard --logdir=runs
     ```
   - Open a browser and go to `http://localhost:6006` to view the TensorBoard dashboard.

5. 模型预测与模拟

    在进行模型预测和模拟之前，请确保已有训练好的模型文件路径正确。可以在 `main.py` 中设置相应路径。

    运行主程序进行预测和模拟：

    ```bash
    python main.py
    ```

    运行后将展示导弹和飞行器的三维轨迹动画。

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

该项目采用 MIT 许可证。详细信息请参阅 LICENSE 文件。

