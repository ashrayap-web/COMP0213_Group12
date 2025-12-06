# Grasp Simulation and Classification Project

A modular PyBullet-based simulation system for robotic grasp experiments with machine learning classifiers.

## Project Structure

```
COMP0213_Group12/
├── main.py                      # Main entry point with CLI
├── objects/                     # Object definitions
│   ├── __init__.py
│   └── sim_objects.py          # SimObject, CubeObject, CylinderObject
├── grippers/                    # Gripper implementations
│   ├── __init__.py
│   ├── base_gripper.py         # SimGripper abstract base class
│   ├── pr2_gripper.py          # PR2 two-finger gripper
│   └── three_finger_hand.py    # SDH three-finger gripper
├── classifiers/                 # ML classifiers
│   ├── __init__.py
│   ├── base_classifier.py      # BaseClassifier abstract class
│   ├── logistic_classifier.py  # Logistic Regression classifier
│   └── mlp_classifier.py       # MLP Neural Network classifier
├── simulation/                  # Simulation logic
│   ├── __init__.py
│   └── grasp_simulation.py     # run_grasp_trial, trainloop, testphaseloop
├── data/                        # CSV training/test data
│   ├── pr2_gripper_cube.csv
│   ├── pr2_gripper_cylinder.csv
│   ├── sdh_gripper_cube.csv
│   └── sdh_gripper_cylinder.csv
└── urdf_files/                  # URDF models
    ├── cylinder.urdf
    ├── sphere.urdf
    └── threeFingers/
        └── sdh/
            ├── sdh.urdf
            └── schunk.material
```

## Installation

### Dependencies

This project requires Python 3.8 or higher.

#### Option 1: Install from requirements.txt (Recommended)

```bash
pip install -r requirements.txt
```

This will install all necessary packages:
- `pybullet` - Physics simulation engine
- `numpy` - Numerical computing
- `pandas` - Data manipulation and CSV handling
- `torch` & `torchvision` - PyTorch for neural network classifier
- `scikit-learn` - Machine learning utilities and logistic regression
- `matplotlib` - Visualization (for bonus extension task)

#### Option 2: Manual installation

```bash
pip install pybullet numpy pandas torch torchvision scikit-learn matplotlib
```

## Usage

This project uses a **command-line interface with argparse** for clear separation of different modes (dataset generation, classifier training, and testing). All parameters are passed as command-line arguments, making it easy to run different experimental configurations.

### Command Structure

The main entry point is `main.py`, which provides two primary modes:
- **`train`** - Generate training dataset through simulation
- **`test`** - Train classifier and test on new grasp trials

### 1. Dataset Generation (Training Mode)

Generate training data through simulation. This mode runs grasp trials and collects position, orientation, and success/failure data.

**Syntax:**
```bash
python main.py train --object <OBJECT_TYPE> --gripper <GRIPPER_TYPE>
```

**Required Arguments:**
- `--object {cube, cylinder}` - Type of object to grasp
- `--gripper {pr2, threefinger}` - Type of gripper to use

**Examples:**
```bash
# PR2 gripper with cylinder
python main.py train --object cylinder --gripper pr2

# PR2 gripper with cube
python main.py train --object cube --gripper pr2

# Three-finger gripper with cube
python main.py train --object cube --gripper threefinger

# Three-finger gripper with cylinder
python main.py train --object cylinder --gripper threefinger
```

**Output:**
- Generates CSV file in `data/` folder (e.g., `data/pr2_gripper_cylinder.csv`)
- CSV contains: x, y, z (position), qx, qy, qz, qw (orientation quaternion), Result (1=success, 0=failure)
- By default, collects 150 successful and 150 failed grasp attempts (300 total samples)

### 2. Classifier Training and Testing (Test Mode)

Train a classifier on existing data and test its predictions on new grasp trials.

**Syntax:**
```bash
python main.py test --object <OBJECT_TYPE> --gripper <GRIPPER_TYPE> --classifier <CLASSIFIER_TYPE>
```

**Required Arguments:**
- `--object {cube, cylinder}` - Type of object to grasp
- `--gripper {pr2, threefinger}` - Type of gripper to use

**Optional Arguments:**
- `--classifier {logistic, mlp}` - Type of classifier (default: mlp)

**Examples:**

**Examples:**
```bash
# Test with MLP classifier (default)
python main.py test --object cylinder --gripper pr2 --classifier mlp

# Test with Logistic Regression classifier
python main.py test --object cube --gripper pr2 --classifier logistic

# Test three-finger gripper with MLP
python main.py test --object cylinder --gripper threefinger --classifier mlp

# Classifier type is optional - defaults to MLP
python main.py test --object cylinder --gripper pr2
```

**What happens in test mode:**
1. Loads existing training data from CSV file
2. Trains the specified classifier (Logistic Regression or MLP)
3. Displays training/validation metrics
4. Runs 10 new grasp trials in simulation
5. Predicts success/failure for each trial
6. Compares predictions against actual outcomes
7. Reports accuracy of predictions

### 3. Getting Help

View available commands and parameters:

```bash
# General help
python main.py --help

# Help for train mode
python main.py train --help

# Help for test mode
python main.py test --help
```

### Command-Line Parameters Summary

**Common Arguments (both modes):**
- `--object {cube, cylinder}` - Object type to grasp
- `--gripper {pr2, threefinger}` - Gripper type to use

**Test Mode Specific:**
- `--classifier {logistic, mlp}` - Classifier algorithm (default: mlp)

**Note:** This project uses **command-line arguments** (via argparse) rather than configuration files. All parameters must be specified when running the script. This design ensures clear, reproducible experiments where all settings are visible in the command itself.

## Available Configurations

### Gripper + Object Combinations

1. **Cylinder + PR2 gripper**
   ```bash
   python main.py test --object cylinder --gripper pr2
   ```

2. **Cube + PR2 gripper**
   ```bash
   python main.py test --object cube --gripper pr2
   ```

3. **Cube + SDH gripper** (three-finger)
   ```bash
   python main.py test --object cube --gripper threefinger
   ```

4. **Cylinder + SDH gripper** (three-finger)
   ```bash
   python main.py test --object cylinder --gripper threefinger
   ```

### Classifier Types

- **Logistic Regression** (`--classifier logistic`): Polynomial features with L2 regularization
- **MLP** (`--classifier mlp`): Multi-layer perceptron neural network

## Module Overview

### Objects (`objects/`)
- `SimObject`: Base class for simulation objects
- `CubeObject`: Cube object for grasping
- `CylinderObject`: Cylinder object with custom friction

### Grippers (`grippers/`)
- `SimGripper`: Abstract base class defining gripper interface
- `PR2Gripper`: Two-finger parallel gripper
- `ThreeFingerHand`: SDH three-finger gripper

### Classifiers (`classifiers/`)
- `BaseClassifier`: Abstract classifier with data loading utilities
- `LogisticRegressionClassifier`: Polynomial logistic regression
- `MLPClassifier`: PyTorch-based neural network

### Simulation (`simulation/`)
- `run_grasp_trial()`: Execute single grasp attempt
- `trainloop()`: Generate training data
- `testphaseloop()`: Train classifier and test predictions

## Data Files

Training data is stored in the `data/` folder with naming convention:
- `{gripper}_{object}.csv`

Each CSV contains columns: `x, y, z, qx, qy, qz, qw, Result`
- Position (x, y, z) and orientation quaternion (qx, qy, qz, qw) of gripper start pose
- Result: 1 (success) or 0 (failure)

## Notes

- The original `combined_classifier.py` is preserved for reference
- URDF files are organized in `urdf_files/` folder
- All paths are configured to work from the project root
- The simulation uses PyBullet's GUI by default

---

## Extension Task: Robotic Arm Reachability Analysis (Bonus Part)

The `robo arm (bonus part)/` folder contains an extension task that analyzes the reachability workspace of a Panda robotic arm using inverse kinematics and machine learning classifiers.

### Overview

This extension explores whether a robotic arm can successfully reach and grasp objects at different positions in 3D space. It generates reachability data through simulation and trains classifiers to predict whether a given (x, y, z) position is reachable.

### Files in Extension Task

```
robo arm (bonus part)/
├── panda_arm1.py                          # Main simulation script for data generation
├── reachability_classifier_LogReg.py      # Logistic Regression classifier
├── reachability_classifier_MLP.py         # MLP Neural Network classifier
├── arm_reachability_down.csv              # Generated reachability dataset
└── best_model.pth                         # Saved MLP model weights
```

### How to Run the Extension Task

#### Step 1: Generate Reachability Dataset

Run the simulation to collect reachability data:

```bash
cd "robo arm (bonus part)"
python panda_arm1.py
```

**What it does:**
- Spawns cube objects at random positions in 3D space
- Attempts to reach each position using inverse kinematics
- Records position (x, y, z) and success/failure (1/0)
- Saves results to `arm_reachability_down.csv`

**Output format:**
- CSV with columns: `x`, `y`, `z`, `Result`
- Result = 1 (reachable), 0 (not reachable)

#### Step 2: Train and Evaluate Classifiers

**Option A: Logistic Regression Classifier**

```bash
python reachability_classifier_LogReg.py
```

This script:
- Loads the reachability dataset
- Visualizes success/failure positions (scatter plot)
- Performs grid search to find optimal hyperparameters
- Trains Logistic Regression with polynomial features
- Displays accuracy, confusion matrix, and classification report
- Shows decision boundary visualization (x-y plane)

**Option B: MLP Neural Network Classifier**

```bash
python reachability_classifier_MLP.py
```

This script:
- Loads the reachability dataset
- Trains a Multi-Layer Perceptron (MLP) neural network
- Shows training progress with loss curves
- Evaluates on test set
- Saves best model to `best_model.pth`
- Generates decision boundary visualization

### Key Differences from Main Task

| Aspect | Main Task | Extension Task |
|--------|-----------|----------------|
| **Problem** | Grasp success prediction | Workspace reachability |
| **Robot** | Grippers (PR2, SDH) | Panda arm |
| **Input Features** | Position + Orientation (7D) | Position only (3D: x, y, z) |
| **Output** | Grasp success/failure | Reachable/unreachable |
| **Method** | End-effector placement | Inverse kinematics |

### Dependencies for Extension Task

The extension task uses the same dependencies as the main project. Ensure you have installed:
```bash
pip install -r requirements.txt
```

Additional visualization will use `matplotlib` (included in requirements).

### Understanding the Results

- **Scatter plots** show the distribution of reachable (green) vs unreachable (red) positions
- **Decision boundary** visualizes the classifier's learned workspace boundary
- **Accuracy metrics** indicate how well the classifier predicts reachability
- The workspace is typically bounded by the arm's joint limits and physical constraints

### Notes on Extension Task

- The Panda arm has 7 degrees of freedom
- Inverse kinematics may have multiple solutions or no solution depending on target position
- The classifiers learn to approximate the complex reachability workspace
- Grid search in LogReg version finds optimal polynomial degree and regularization
- MLP version can capture non-linear workspace boundaries effectively
