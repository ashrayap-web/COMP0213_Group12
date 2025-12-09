"""Grasp simulation functions"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import pandas as pd

from objects import CubeObject, CylinderObject
from grippers import PR2Gripper, ThreeFingerHand
from classifiers import LogisticRegressionClassifier, MLPClassifier


def run_grasp_trial(i, object_choice="cylinder", gripper_choice="pr2", gripper_urdf="pr2_gripper.urdf"):
    """
    Runs a single grasp attempt and returns the outcome together with the
    configured gripper and object so callers can reuse their state.
    """
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)
    p.loadURDF("plane.urdf")

    # Choose the object
    if object_choice == "cube":
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        current_obj = CubeObject(
            f"Cube{i+1}",
            pos=[0.0, 0.0, 0.06],
            orientation=cube_start_orientation
        )
    else:
        cylinder_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        current_obj = CylinderObject(
            f"Cylinder{i+1}",
            pos=[0.0, 0.0, 0.06],
            orientation=cylinder_start_orientation
        )

    for _ in range(50): 
        p.stepSimulation()
        #time.sleep(1./240.)

    # Choose the gripper
    if gripper_choice == "threefinger":
        curr_gripper = ThreeFingerHand(gripper_urdf, pos=None, orientation=None, target_obj=current_obj)
    else:
        curr_gripper = PR2Gripper(gripper_urdf, pos=None, orientation=None, target_obj=current_obj)
    
    curr_gripper.open_gripper()
    for _ in range(50): 
        p.stepSimulation()
        #time.sleep(1./240.)

    curr_gripper.move_towards_obj()
    for _ in range(80): 
        p.stepSimulation()
        #time.sleep(1./240.)

    curr_gripper.close_gripper()
    for _ in range(750): 
        p.stepSimulation()
        #time.sleep(1./240.)

    current_obj.pos_grab_before, _ = p.getBasePositionAndOrientation(current_obj.body_id)

    curr_gripper.grab_start_pos, _ = p.getBasePositionAndOrientation(curr_gripper.body_id)
    x, y, z = curr_gripper.grab_start_pos
    curr_gripper.move_gripper(x, y, z + 0.3)
    for _ in range(50): 
        p.stepSimulation()
        #time.sleep(1./240.)

    for _ in range(120): 
        p.stepSimulation()
        #time.sleep(1./240.)

    current_obj.pos_grab_after, _ = p.getBasePositionAndOrientation(current_obj.body_id)
    curr_gripper.grab_end_pos, _ = p.getBasePositionAndOrientation(curr_gripper.body_id)

    result = curr_gripper.is_success()
    for _ in range(50): 
        p.stepSimulation()
        #time.sleep(1./240.)

    return result, curr_gripper, current_obj


def trainloop(object_choice="cylinder", gripper_choice="pr2", gripper_urdf="pr2_gripper.urdf", 
              csvfile="data/pr2_gripper_cylinder.csv"):
    """Generate training data through simulation"""
    # Setup
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0.2])
    p.setPhysicsEngineParameter(numSolverIterations=300, erp=0.3, contactERP=0.3)

    n = 10000
    target_samples = 200
    count1 = 0
    count0 = 0
    data_rows = []

    for i in range(n):
        result, curr_gripper, current_obj = run_grasp_trial(
            i,
            object_choice=object_choice,
            gripper_choice=gripper_choice,
            gripper_urdf=gripper_urdf
        )
        if result == 1 and count1 < target_samples:
            count1 += 1
            data_rows.append(np.hstack([curr_gripper.start_pos, curr_gripper.orientation, result]))
        elif result == 0 and count0 < target_samples:
            count0 += 1
            data_rows.append(np.hstack([curr_gripper.start_pos, curr_gripper.orientation, result]))
        if count1 == target_samples and count0 == target_samples:
            break

    p.disconnect()
    cols = ["x", "y", "z", "qx", "qy", "qz", "qw", "Result"]
    df = pd.DataFrame(data_rows, columns=cols)
    print(df.head())
    df.to_csv(csvfile, index=False)
    print(f"\nTraining data saved to {csvfile}")


def testphaseloop(classifier_type="logistic", object_choice="cylinder", 
                  gripper_choice="pr2", gripper_urdf="pr2_gripper.urdf", 
                  csvfile="data/pr2_gripper_cylinder.csv"):
    """Test the classifier on new grasp trials
    
    Args:
        classifier_type: "logistic" or "mlp" to choose classifier type
    """
    # Setup
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0.2])
    p.setPhysicsEngineParameter(numSolverIterations=300, erp=0.3, contactERP=0.3)

    data_rows = []
    prediction_results = []
    
    # Choose classifier based on type
    if classifier_type.lower() == "logistic":
        print("\n=== Using Logistic Regression Classifier ===")
        classifier = LogisticRegressionClassifier(csvfile=csvfile)
    elif classifier_type.lower() == "mlp":
        print("\n=== Using MLP Neural Network Classifier ===")
        classifier = MLPClassifier(csvfile=csvfile)
    else:
        raise ValueError("classifier_type must be 'logistic' or 'mlp'")
    
    # Train the classifier
    classifier.train()

    # Test on 10 new grasps
    for i in range(10):
        result, curr_gripper, current_obj = run_grasp_trial(
            i,
            object_choice=object_choice,
            gripper_choice=gripper_choice,
            gripper_urdf=gripper_urdf
        )
        grasp_features = np.hstack([curr_gripper.start_pos, curr_gripper.orientation])
        predicted_label = float(classifier.predict(grasp_features.reshape(1, -1))[0])
        classifier_correct = predicted_label == result
        prediction_results.append({
            "grasp": i + 1,
            "predicted": predicted_label,
            "actual": result,
            "correct": classifier_correct
        })
        print(
            f"Classifier prediction for grasp {i + 1}: "
            f"{'correct' if classifier_correct else 'incorrect'} "
            f"(pred={predicted_label}, actual={result})"
        )
        data_rows.append(np.hstack([grasp_features, result]))

    p.disconnect()
    cols = ["x", "y", "z", "qx", "qy", "qz", "qw", "Result"]
    df = pd.DataFrame(data_rows, columns=cols)
    print(df.head())
    total_correct = sum(1 for entry in prediction_results if entry["correct"])
    print(f"\nClassifier matched {total_correct} / {len(prediction_results)} grasps.")
