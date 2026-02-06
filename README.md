# Disentangeld Shared-Specific Dynamics Learning for Cross-System Identification

## Additional Files

Other project files are available at: [https://drive.google.com/drive/folders/1UlD6YIVsCEYH2_ZC5PC16jn9Kqm4-Q5L?usp=sharing](https://drive.google.com/drive/folders/1UlD6YIVsCEYH2_ZC5PC16jn9Kqm4-Q5L?usp=sharing). Please download them yourself.

## Project Introduction

This project implements a deep learning-based sparse system identification method, specifically for pendulum system dynamics modeling. Through meta-learning and sparse parameter learning techniques, this method can:

- Learn shared dynamical representations from multiple similar systems
- Automatically discover key system parameters using sparse masks
- Quickly adapt to new unknown systems with only a small amount of data
- Improve model interpretability and generalization capabilities

## Technology Stack

- **Programming Language**: Python 3.9+
- **Deep Learning Framework**: PyTorch
- **Data Processing**: NumPy, CSV
- **Visualization**: Matplotlib
- **Optimizers**: Adam, LBFGS, custom CautiousAdam

## Project Structure

```
├── datasets/           # Dataset directory
│   ├── pendulum_real/  # Real pendulum system data
│   ├── __init__.py     # Dataset initialization
│   └── utils.py        # Data processing utilities
├── draw/               # Plotting scripts
├── final_data/         # Final result data
│   ├── hidden_params/  # Learned hidden parameters
│   ├── mask/           # Sparse mask data
│   ├── meta_train/     # Meta-training loss data
│   └── pretrain/       # Pretraining loss data
├── model/              # Model saving directory
├── pictures/           # Result pictures
├── results/            # Evaluation results
├── DSSDL_mask.py       # Core model implementation
├── evaluate_mask.py    # Mask evaluation
├── fast_adapt.py       # Fast adaptation script
├── meta_train.py       # Meta-training script
└── README.md           # Project description
```

## Core Algorithms

### 1. Meta-learning

The meta-learning phase trains the model on multiple pendulum systems to learn shared dynamical representations. Key steps include:

- **Multi-system training**: Training with data from multiple pendulum systems with different parameters
- **Parameter sharing**: Learning shared features and parameters across systems
- **Sparse regularization**: Encouraging parameter sparsity through masking mechanisms

### 2. Sparse Parameter Learning

Using learnable masks to automatically discover key system parameters:

- **Continuous mask**: Sigmoid-activated continuous value mask
- **Hard mask**: Threshold-based binary mask for true sparsity
- **Mask regularization**: Encouraging mask values to approach 0 or 1 for enhanced sparsity

### 3. Fast Adaptation

Using meta-learned initialization to quickly adapt to new systems with only a small amount of support data:

- **Parameter fine-tuning**: Only fine-tuning parameters specific to the new system
- **Early stopping**: Stopping training when loss change falls below a threshold
- **Fast convergence**: Leveraging good initialization from meta-learning to accelerate convergence

## Model Architecture

### MetaNet Model

MetaNet is the core model of this project, consisting of the following components:

1. **One-hot to specific parameter mapping**: Maps system one-hot encoding to specific parameters
2. **Continuous mask layer**: Learns sparsity mask for parameters
3. **Neural network backbone**: Composed of multiple fully connected layers and SiLU activation functions
4. **Specific parameter generator**: Generates specific parameters for new systems during fast adaptation

```python
# MetaNet model architecture
MetaNet(
  (one_hot_to_specific_param): Linear(in_features=N, out_features=D, bias=False)
  (specific_param_generator): Linear(in_features=1, out_features=D, bias=False)
  (continuous_mask): Parameter(torch.Size([D]))
  (backbone): Sequential(
    (0): Linear(in_features=I+D, out_features=64)
    (1): SiLU()
    (2): Linear(in_features=64, out_features=64)
    (3): SiLU()
    (4): Linear(in_features=64, out_features=O)
  )
)
```

Where:
- N: Number of training systems
- D: Dimension of specific parameters
- I: Dimension of input states
- O: Dimension of output derivatives

## Training Process

### 1. Meta-training

```bash
python meta_train.py
```

Meta-training process:
1. Load training data from multiple pendulum systems
2. Initialize the MetaNet model
3. Perform meta-training to learn shared representations and sparse masks
4. Save the trained model and loss curves

### 2. Fast Adaptation

```bash
python fast_adapt.py
```

Fast adaptation process:
1. Load the meta-trained model
2. Quickly fine-tune the model using support set data
3. Evaluate model performance on the query set
4. Save the adapted model and results

## Configuration Parameters

### Meta-training Configuration

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `specific_param_dim` | Dimension of specific (slow) parameters | 6 |
| `epochs` | Number of meta-training epochs | 100 |
| `reg_rate` | Regularization rate | 0.0 |
| `use_raw_data` | Whether to use raw (unnormalized) data | False |
| `use_trajectory_sampling` | Whether to use trajectory sampling | False |
| `num_trajectories` | Number of trajectories to sample per system | 1 |
| `trajectory_duration` | Duration of each trajectory (seconds) | 10 |
| `batch_size` | Batch size | 128 |
| `meta_train_lr` | Meta-training learning rate | 0.01 |
| `hidden_widths` | Hidden layer widths | [64, 64] |
| `fast_adapt_lr` | Fast adaptation learning rate | 0.001 |
| `optimizer` | Optimizer type | 'adam' |
| `device` | Device to use | 'cuda:0' |

### Fast Adaptation Configuration

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `epochs` | Number of fast adaptation epochs | 200 |
| `convergence_threshold` | Convergence threshold | 1e-5 |
| `patience` | Early stopping patience | 100 |
| `batch_size` | Batch size | -1 (use full batch) |
| `optimizer` | Optimizer type | 'lbfgs' |

## Dataset Format

The dataset format used in this project is as follows:

```
datasets/
└── pendulum_real/
    ├── train/         # Training set (multiple systems)
    │   ├── features/  # State features
    │   └── labels/    # Derivative labels
    ├── support/       # Support set (new system)
    │   ├── features/  # State features
    │   └── labels/    # Derivative labels
    └── query/         # Query set (new system)
        ├── features/  # State features
        └── labels/    # Derivative labels
```

Each system's data is stored as text files, where:
- Feature files contain the system's state variables (e.g., angle, angular velocity)
- Label files contain the derivatives of the state variables

## Result Analysis

### 1. Loss Curves

Loss curves during training are saved in the `pictures/` directory, including:
- Meta-training loss curve (`meta_train_loss_curve.png`)
- Pretraining loss curve (`pretrain_loss_curve.png`)
- Fast adaptation loss curve

### 2. Sparse Mask Analysis

Learned sparse masks are saved in the `final_data/mask/` directory, and system key parameters can be analyzed through mask values.

### 3. Trajectory Prediction

Model trajectory prediction results on new systems are saved in the `pictures/` directory, such as `trajectory_pendulum_real.png`.

### 4. Hidden Parameter Analysis

Learned system hidden parameters are saved in the `final_data/hidden_params/` directory, which can be used to analyze the physical characteristics of the system.

## Technical Highlights

1. **Sparse Parameter Learning**: Automatically discovering system key parameters through learnable masks, improving model interpretability.

2. **Meta-learning**: Learning shared representations from multiple systems, improving model generalization ability.

3. **Fast Adaptation**: Quickly adapting to new systems with only a small amount of data, reducing data requirements.

4. **Custom Optimizer**: Implementing the CautiousAdam optimizer that adjusts learning rate based on gradient direction, improving training stability.

5. **Trajectory Sampling**: Supporting trajectory-based sampling methods to better capture system dynamic characteristics.

## Experimental Results

Experimental results on pendulum systems show:

- Meta-learning significantly improves model generalization ability
- Sparse masks successfully discover system key parameters
- Fast adaptation achieves good performance with only a small amount of data
- The model has higher prediction accuracy on unknown systems compared to traditional methods

## Code Usage Guide

### 1. Prepare Dataset

Organize pendulum system data into the `datasets/` directory according to the format described above.

### 2. Meta-training

```bash
# Execute meta-training
python meta_train.py

# Meta-training results will be saved to:
# - model/meta_train.pt (model weights)
# - final_data/meta_train/meta_train_loss.npy (loss data)
# - pictures/pendulum_real/meta_train_loss_curve.png (loss curve)
```

### 3. Fast Adaptation

```bash
# Execute fast adaptation
python fast_adapt.py

# Fast adaptation results will be saved to:
# - model/fast_adapt.pt (adapted model)
# - model/fast_adapt_loss.npy (adaptation loss)
# - results/pendulum_real/ (evaluation results)
```

### 4. Custom Configuration

You can adjust model parameters and training settings by modifying the configuration dictionaries in `meta_train.py` and `fast_adapt.py`.

## Extension and Application

This method is not only applicable to pendulum systems but can also be extended to other types of dynamical systems, such as:

- Spring-mass systems
- Robot dynamics models
- Chemical reaction dynamics
- Financial time series modeling

By adjusting the model's input dimension and hidden layer widths, it can adapt to systems of different complexities.

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- NumPy
- Matplotlib
- tqdm

## References

1. Meta-Learning for Few-Shot Learning. Chelsea Finn, Pieter Abbeel, Sergey Levine. 2017.
2. Sparse Identification of Nonlinear Dynamical Systems. Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz. 2016.
3. Deep Sparse Regularization for Neural Networks. Zhang et al. 2018.

## Code Description

### Core Modules

1. **DSSDL_mask.py**: Implements the core MetaNet model and trainer class, including sparse mask learning and meta-learning functionality.

2. **meta_train.py**: Implements the meta-training process, training the model on multiple systems.

3. **fast_adapt.py**: Implements the fast adaptation process, quickly adapting to new systems with small amounts of data.

4. **evaluate_mask.py**: Used for evaluating and analyzing learned sparse masks.

### Data Processing

Dataset loading and processing functionality is located in the `datasets/` directory, supporting:
- Data normalization
- Trajectory sampling
- One-hot encoding
- Batch loading

## Troubleshooting

1. **CUDA Out of Memory**:
   - Reduce batch size `batch_size`
   - Use smaller hidden layer widths `hidden_widths`
   - Switch to CPU device `device='cpu'`

2. **Training Instability**:
   - Adjust learning rates `meta_train_lr` and `fast_adapt_lr`
   - Use CautiousAdam optimizer `use_cautious_adam=True`
   - Increase regularization rate `reg_rate`

3. **Poor Fast Adaptation Performance**:
   - Increase support set data amount
   - Adjust `convergence_threshold` and `patience` parameters
   - Extend fast adaptation epochs `epochs`

## Future Work

1. **Extension to More System Types**: Apply the method to a wider range of dynamical systems.

2. **Improved Mask Learning**: Develop more effective sparse mask learning methods.

3. **Physical Constraint Integration**: Integrate physical laws as constraints into the model.

4. **Uncertainty Estimation**: Add uncertainty estimation functionality to improve model reliability.

5. **Real-time Applications**: Optimize the model to support real-time system identification and control.

## Contact

If you have any questions or suggestions, please feel free to contact the project maintainers.

---

**Project Version**: 1.0.0
**Last Updated**: 2026-02-06
**License**: MIT
