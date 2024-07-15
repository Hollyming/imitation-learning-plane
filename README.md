# Behavior Cloning Project

This project is for reinforce learning in imitation learing area. We use ourselves dataset to genertate the original agent. All the project is responsible to SJTU, China ShangHai BATC lab.

This repository contains the implementation of behavior cloning for agent training using different network architectures and monitoring tools. The project includes three main scripts:

1. `bc.py`: The first version of behavior cloning code.
2. `bc_2.py`: The second version with TensorBoard monitoring.
3. `bc_trans.py`: Utilizing Transformer as the backbone.

## Project Structure

```
├── bc.py
├── bc_2.py
├── bc_trans.py
├── README.md
└── data
    └── hit_false
```

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

3. **Run the scripts:**
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

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
