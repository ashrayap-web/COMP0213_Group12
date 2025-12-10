#!/usr/bin/env python
"""
Main entry point for the grasp simulation and classification experiments.

Usage examples:
    # Generate training data for PR2 gripper with cylinder
    python main.py train --object cylinder --gripper pr2
    
    # Test MLP classifier on cube grasping with PR2
    python main.py test --object cube --gripper pr2 --classifier mlp
    
    # Test with three-finger gripper on cylinder
    python main.py test --object cylinder --gripper threefinger --classifier logistic
"""

import argparse
import sys
from simulation import trainloop, testphaseloop


# Configuration mapping for different gripper and object combinations
GRIPPER_CONFIGS = {
    "pr2": {
        "urdf": "pr2_gripper.urdf",
        "choice": "pr2"
    },
    "threefinger": {
        "urdf": "./urdf_files/threeFingers/sdh/sdh.urdf",
        "choice": "threefinger"
    }
}

OBJECT_CONFIGS = {
    "cube": "cube",
    "cylinder": "cylinder"
}


def get_csv_filename(gripper, obj):
    """Generate CSV filename based on gripper and object combination"""
    gripper_name = "sdh_gripper" if gripper == "threefinger" else "pr2_gripper"
    return f"data/{gripper_name}_{obj}.csv"


def train_command(args):
    """Execute training loop to generate grasp data"""
    gripper_config = GRIPPER_CONFIGS[args.gripper]
    csvfile = get_csv_filename(args.gripper, args.object)
    
    print(f"\n{'='*60}")
    print(f"TRAINING MODE: Generating grasp data")
    print(f"{'='*60}")
    print(f"Object: {args.object}")
    print(f"Gripper: {args.gripper}")
    print(f"Output CSV: {csvfile}")
    print(f"{'='*60}\n")
    
    trainloop(
        object_choice=args.object,
        gripper_choice=gripper_config["choice"],
        gripper_urdf=gripper_config["urdf"],
        csvfile=csvfile
    )


def test_command(args):
    """Execute test loop with classifier"""
    gripper_config = GRIPPER_CONFIGS[args.gripper]
    csvfile = get_csv_filename(args.gripper, args.object)
    
    # Use special dataset for PR2 + cube + MLP combination as it requires more dat.
    if args.gripper == "pr2" and args.object == "cube" and args.classifier == "mlp":
        csvfile = "data/pr2_gripper_cube_mlp.csv"
    
    print(f"\n{'='*60}")
    print(f"TEST MODE: Evaluating classifier")
    print(f"{'='*60}")
    print(f"Object: {args.object}")
    print(f"Gripper: {args.gripper}")
    print(f"Classifier: {args.classifier}")
    print(f"Training CSV: {csvfile}")
    print(f"{'='*60}\n")
    
    testphaseloop(
        classifier_type=args.classifier,
        object_choice=args.object,
        gripper_choice=gripper_config["choice"],
        gripper_urdf=gripper_config["urdf"],
        csvfile=csvfile
    )


def main():
    parser = argparse.ArgumentParser(
        description="Grasp Simulation and Classification Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with different configurations
  python main.py train --object cylinder --gripper pr2
  python main.py train --object cube --gripper threefinger
  
  # Test with different classifiers
  python main.py test --object cylinder --gripper pr2 --classifier logistic
  python main.py test --object cube --gripper pr2 --classifier mlp
  python main.py test --object cylinder --gripper threefinger --classifier mlp
  
Available combinations:
  Objects: cube, cylinder
  Grippers: pr2, threefinger
  Classifiers: logistic, mlp
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Generate training data through simulation")
    train_parser.add_argument(
        "--object",
        choices=["cube", "cylinder"],
        required=True,
        help="Object type to grasp"
    )
    train_parser.add_argument(
        "--gripper",
        choices=["pr2", "threefinger"],
        required=True,
        help="Gripper type to use"
    )
    train_parser.set_defaults(func=train_command)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test classifier on grasp trials")
    test_parser.add_argument(
        "--object",
        choices=["cube", "cylinder"],
        required=True,
        help="Object type to grasp"
    )
    test_parser.add_argument(
        "--gripper",
        choices=["pr2", "threefinger"],
        required=True,
        help="Gripper type to use"
    )
    test_parser.add_argument(
        "--classifier",
        choices=["logistic", "mlp"],
        default="mlp",
        help="Classifier type (default: mlp)"
    )
    test_parser.set_defaults(func=test_command)
    
    # Parse arguments and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
