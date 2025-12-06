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

Install required dependencies:

```bash
pip install pybullet numpy pandas torch scikit-learn
```

## Usage

The project uses a command-line interface with argparse for easy experimentation.

### Training Mode

Generate training data through simulation:

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

### Test Mode

Train a classifier and test on new grasp trials:

```bash
# Test with MLP classifier (default)
python main.py test --object cylinder --gripper pr2 --classifier mlp

# Test with Logistic Regression classifier
python main.py test --object cube --gripper pr2 --classifier logistic

# Test three-finger gripper with MLP
python main.py test --object cylinder --gripper threefinger --classifier mlp
```

### Command-Line Options

**Common arguments:**
- `--object {cube,cylinder}` - Object type to grasp
- `--gripper {pr2,threefinger}` - Gripper type to use

**Test-specific arguments:**
- `--classifier {logistic,mlp}` - Classifier type (default: mlp)

### Get Help

```bash
python main.py --help
python main.py train --help
python main.py test --help
```

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
