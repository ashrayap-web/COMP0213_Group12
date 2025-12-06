import pybullet as p
import pybullet_data
import time
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from abc import ABC, abstractmethod


class SimObject:
    def __init__(self, name, urdf_file=None, pos=None, orientation=None, scale=1.0):
        self.name = name
        self.pos = pos
        self.orientation = orientation
        self.body_id = p.loadURDF(urdf_file, basePosition=self.pos, baseOrientation=self.orientation, globalScaling=scale)
        self.pos_grab_before = None
        self.pos_grab_after = None

class CubeObject(SimObject):
    def __init__(self, name, urdf_file="cube_small.urdf", pos=None, orientation=None, scale=1.0):
        super().__init__(name, urdf_file, pos, orientation, scale)

class CylinderObject(SimObject):
    def __init__(self, name, urdf_file="cylinder.urdf", pos=None, orientation=None, scale=0.55):
        super().__init__(name, urdf_file, pos, orientation, scale)
        p.changeDynamics(self.body_id, -1, lateralFriction=1.0)

# ---------------------------------------------------------
# GRIPPER INTERFACE
# ---------------------------------------------------------

class SimGripper(ABC):
    def __init__(self, pos=None, target_obj=None):
        self.OBJ = target_obj
        self.start_pos = np.array(pos if pos is not None else [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.6, 1)], dtype=float)
        self.grab_start_pos = None
        self.grab_end_pos = None
    @abstractmethod
    def set_orientation(self):
        pass
    
    @abstractmethod
    def open_gripper(self):
        pass
    
    @abstractmethod
    def close_gripper(self):
        pass
    
    @abstractmethod
    def move_gripper(self, x, y, z, force=80):
        pass
    @abstractmethod
    def move_towards_obj(self):
        pass
    def set_orientation(self):
        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        d_vector = np.array(obj_pos) - self.start_pos 
        dx, dy, dz = d_vector
        if np.abs(dx) < 1e-3 and np.abs(dy) < 1e-3:
            roll, yaw, pitch = -np.pi/2, 0, np.pi/2
        else:
            pitch  = np.atan2(-dz, np.sqrt(dx**2 + dy**2)) 
            yaw = np.atan2(dy, dx)
            roll = 0
            
        # yaw += random.uniform(-np.pi/36, np.pi/36) 
        # pitch += random.uniform(-np.pi/36, np.pi/36)
        yaw += np.random.normal(0, np.pi/72)
        pitch += np.random.normal(0, np.pi/72)
        return p.getQuaternionFromEuler([roll, pitch, yaw])
    def is_success(self):
        start_pos_obj = np.array(self.OBJ.pos_grab_before)
        end_pos_obj = np.array(self.OBJ.pos_grab_after)
        dist_diff_vec = (np.array(self.grab_end_pos) - np.array(self.grab_start_pos)) - (end_pos_obj - start_pos_obj)
        _, angular_vel = p.getBaseVelocity(self.OBJ.body_id)
        if dist_diff_vec[2] < 0.01 and np.linalg.norm(angular_vel) < 0.25:
            print(f"\033[32m{self.OBJ.name}: SUCCESS\033[0m")
            return 1.0
        else:
            print(f"\033[31m{self.OBJ.name}: FAILURE\033[0m")
            return 0.0


# ---------------------------------------------------------
# PR2 GRIPPER (two finger)
# ---------------------------------------------------------
class PR2Gripper(SimGripper):
    def __init__(self, urdf_file, pos=None, orientation=None, target_obj=None):
        super().__init__(pos, target_obj)
        self.orientation = np.array(orientation if orientation is not None else self.set_orientation(), dtype=float)
        self.body_id = p.loadURDF(urdf_file, basePosition=self.start_pos.tolist(), baseOrientation=self.orientation.tolist())
        self.cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0,0,0], self.start_pos.tolist(), [0,0,0,1], self.orientation.tolist())

    def close_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.1, maxVelocity=2, force=10)

    def open_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.95, maxVelocity=2, force=10)

    def move_gripper(self, x, y, z, force=80):
        p.changeConstraint(self.cid, jointChildPivot=[x, y, z], jointChildFrameOrientation = self.orientation, maxForce=force)

    def move_towards_obj(self):
        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        obj_pos = np.array(obj_pos); obj_pos[2] += 0.005
        curr_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        d_vec = obj_pos - np.array(curr_pos)
        pos_step = obj_pos - 0.30 * (d_vec / np.linalg.norm(d_vec))
        self.move_gripper(pos_step[0], pos_step[1], pos_step[2], force=1000)

# ---------------------------------------------------------
# ThreeFingerHand gripper
# ---------------------------------------------------------
class ThreeFingerHand(SimGripper):
    GRASP_JOINTS = [1, 4, 7]
    PRESHAPE_JOINTS = [2, 5, 8]
    UPPER_JOINTS = [3, 6, 9]

    def __init__(self, urdf_file, pos=None, orientation=None, target_obj=None):
        super().__init__(pos, target_obj)
        base_orientation_quat = self.set_orientation()
        TWIST_ANGLE = np.pi / 2 
        twist_quat = p.getQuaternionFromEuler([np.pi/2, 0, TWIST_ANGLE])
        twisted_orientation_quat = p.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=base_orientation_quat,
            positionB=[0, 0, 0],
            orientationB=twist_quat,
        )[1] 
        self.orientation = np.array(orientation if orientation is not None else twisted_orientation_quat, dtype=float)
        self.body_id = p.loadURDF(urdf_file, basePosition=self.start_pos.tolist(), baseOrientation=self.orientation.tolist(), globalScaling=1.0)
        self.gripper_id = self.body_id 
        self.open = False
        self.num_joints = p.getNumJoints(self.body_id)
        
        # Filter the lists. If joint index >= num_joints, remove it.
        self.GRASP_JOINTS = [j for j in self.GRASP_JOINTS if j < self.num_joints]
        self.PRESHAPE_JOINTS = [j for j in self.PRESHAPE_JOINTS if j < self.num_joints]
        self.UPPER_JOINTS = [j for j in self.UPPER_JOINTS if j < self.num_joints]
        # ----------------------------------------------------------

        # Movement Constraint
        self.cid = p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0,0,0],
            childFramePosition= self.start_pos.tolist(),
            parentFrameOrientation = [0,0,0,1],
            childFrameOrientation = self.orientation.tolist()
        )

    def preshape(self):
        """Move fingers into preshape pose."""
        for i in [2, 5, 8]:
            p.setJointMotorControl2(self.gripper_id, i, p.POSITION_CONTROL,
                                    targetPosition=-0.7, maxVelocity=0.5, force=1)
        self.open = False
        #time.sleep(1)
        
        

    def open_gripper(self):
        """Gradually open fingers until fully open."""
        closed, iteration = True, 0
        while closed and not self.open:
            joints = self.get_joint_positions()
            closed = False
            for k in range(self.num_joints):
                if k in self.PRESHAPE_JOINTS and joints[k] >= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.01)
                    closed = True
                elif k in self.UPPER_JOINTS and joints[k] <= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.01)
                    closed = True
                elif k in self.GRASP_JOINTS and joints[k] <= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.01)
                    closed = True
            iteration += 1
            if iteration > 1000: # Reduced iteration limit for speed
                break
            p.stepSimulation()
        self.open = True

    def _apply_joint_command(self, joint, target):
        p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                targetPosition=target, maxVelocity=2, force=9999)

    def get_joint_positions(self):
        return [p.getJointState(self.gripper_id, i)[0] for i in range(self.num_joints)]

    def close_gripper(self): 
        "Close gripper to grab object"
        self._apply_joint_command(
            joint=7,
            target=-0.5)
        for j in [1, 4, 7]:
            self._apply_joint_command(
                joint=j, target=0.3)
        #time.sleep(2)
        self.open=False
        
    def move_gripper(self, x, y, z, force=80):
        p.changeConstraint(
            self.cid,
            jointChildPivot=[x, y, z],
            jointChildFrameOrientation = self.orientation,
            maxForce=force
        )

    def move_towards_obj(self):
        min_dist = 0.17 #0.17 with cylinder
        z_offset = 0  #0 with cylinder
        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        obj_pos = np.array(obj_pos); obj_pos[2] += z_offset
        curr_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        d_vec = obj_pos - np.array(curr_pos)
        pos_step = obj_pos - min_dist *(d_vec / np.linalg.norm(d_vec)) 
        self.move_gripper(pos_step[0], pos_step[1], pos_step[2], force=1000)


# ---------------------------------------------------------
# BASE CLASSIFIER CLASS (Abstract)
# ---------------------------------------------------------
class BaseClassifier(ABC):
    """Base classifier class for grasp success prediction"""
    
    def __init__(self, csvfile="pr2_gripper_cylinder.csv"):
        self.csvfile = csvfile
        self.model = None
        
    def load_and_balance_data(self):
        """Load CSV and balance the dataset"""
        df = pd.read_csv(self.csvfile)
        min_n = df["Result"].value_counts().min()
        print(f"Balancing dataset: {min_n} samples per class")
        balanced_df = df.groupby("Result", group_keys=False).sample(n=min_n, random_state=42)
        
        X = balanced_df.drop(columns=["Result"])
        y = balanced_df["Result"]
        
        return X, y
    
    @abstractmethod
    def train(self):
        """Train the classifier model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Predict class labels for samples in X"""
        pass
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        y_pred = self.predict(X_test)
        
        print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Confusion Matrix\n", confusion_matrix(y_test, y_pred), "\n")
        print("Classification Report\n", classification_report(y_test, y_pred), "\n")


# ---------------------------------------------------------
# LOGISTIC REGRESSION CLASSIFIER
# ---------------------------------------------------------
class LogisticRegressionClassifier(BaseClassifier):
    """Logistic Regression classifier with polynomial features"""
    
    def __init__(self, csvfile="pr2_gripper_cylinder.csv", degree=6, C=10.0):
        super().__init__(csvfile)
        self.degree = degree
        self.C = C
        
    def train(self):
        """Train logistic regression model"""
        X, y = self.load_and_balance_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Create pipeline with polynomial features and logistic regression
        self.model = make_pipeline(
            PolynomialFeatures(degree=self.degree),
            StandardScaler(),
            LogisticRegression(penalty="l2", C=self.C, max_iter=10000)
        )
        
        print("\nTraining Logistic Regression classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on train and test
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
        print("Test Accuracy:", accuracy_score(y_test, y_test_pred), "\n")
        
        print("Confusion Matrix\n", confusion_matrix(y_test, y_test_pred), "\n")
        print("Classification Report\n", classification_report(y_test, y_test_pred), "\n")
        
        return self.model
    
    def predict(self, X):
        """Predict class labels"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)


# ---------------------------------------------------------
# MLP CLASSIFIER
# ---------------------------------------------------------
class MLP(nn.Module):
    """Multi-Layer Perceptron for grasp classification"""
    def __init__(self, input_dim):
        super().__init__()
        dropout = 0.15
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class MLPClassifier(BaseClassifier):
    """MLP Neural Network classifier"""
    
    def __init__(self, csvfile="pr2_gripper_cylinder.csv", epochs=300, batch_size=64, lr=0.001):
        super().__init__(csvfile)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = None
        self.nn_model = None
        
    def train(self):
        """Train MLP classifier"""
        X, y = self.load_and_balance_data()
        
        # Create train, val and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        # Feature normalization
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        
        # Create data loaders
        train_data = TensorDataset(X_train_t, y_train_t)
        val_data = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        self.nn_model = MLP(X_train_t.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Training loop
        print("\nTraining MLP classifier...")
        
        for epoch in range(self.epochs):
            self.nn_model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.nn_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * y_batch.size(0)
            
            # Validation
            self.nn_model.eval()
            correct_preds = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.nn_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * y_batch.size(0)
                    preds_cls = (outputs > 0.5).float()
                    correct_preds += preds_cls.eq(y_batch).sum().item()
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch: {(epoch+1)}/{self.epochs}   "
                      f"Train Loss: {train_loss/len(train_loader.dataset):.4f}   "
                      f"Val Accuracy: {100*correct_preds/len(val_loader.dataset):.2f}% -> {correct_preds}/{len(val_loader.dataset)}   "
                      f"Val Loss: {val_loss/len(val_loader.dataset):.4f}")
        
        # Test accuracy
        self.nn_model.eval()
        with torch.no_grad():
            preds = self.nn_model(X_test_t)
            preds_class = (preds > 0.5).float()
            accuracy = (preds_class.eq(y_test_t).sum() / y_test_t.shape[0]).item()
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Convert to numpy for metrics
        y_test_np = y_test_t.numpy()
        y_pred_np = preds_class.numpy()
        
        print("Confusion Matrix\n", confusion_matrix(y_test_np, y_pred_np), "\n")
        print("Classification Report\n", classification_report(y_test_np, y_pred_np), "\n")
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if self.nn_model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.nn_model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.nn_model(X_tensor)
            predictions = (outputs > 0.5).float().numpy().flatten()
        
        return predictions


# ---------------------------------------------------------
# SIMULATION FUNCTIONS
# ---------------------------------------------------------
def run_grasp_trial(i, object_choice="cylinder",
                    gripper_choice="pr2", gripper_urdf="pr2_gripper.urdf"):
    """
    Runs a single grasp attempt and returns the outcome together with the
    configured gripper and object so callers can reuse their state.
    """
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)
    p.loadURDF("plane.urdf")

    #----choose the object you want---#
    
    if object_choice == "cube":
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        current_obj = CubeObject(
            f"Cube{i+1}",
            
            pos=[0.0,0.0, 0.06],
            orientation=cube_start_orientation
        )
    else:
        cylinder_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        current_obj = CylinderObject(
            f"Cylinder{i+1}",
           
            pos=[0.0,0.0, 0.06],
            orientation=cylinder_start_orientation
        )
    #---------------------------------#
    

    for _ in range(50): p.stepSimulation(); time.sleep(1./240.)

    #-----choose the gripper you want------#
    if gripper_choice == "threefinger":
        curr_gripper = ThreeFingerHand(gripper_urdf, pos=None, orientation=None, target_obj=current_obj)
    else:
        curr_gripper = PR2Gripper(gripper_urdf, pos=None, orientation=None, target_obj=current_obj)
    #--------------------------------------#
    
    curr_gripper.open_gripper()
    for _ in range(50): p.stepSimulation(); time.sleep(1./240.)

    curr_gripper.move_towards_obj()
    for _ in range(80): p.stepSimulation(); time.sleep(1./240.)

    curr_gripper.close_gripper()
    for _ in range(350): p.stepSimulation(); time.sleep(1./240.)

    current_obj.pos_grab_before, _ = p.getBasePositionAndOrientation(current_obj.body_id)

    curr_gripper.grab_start_pos, _ = p.getBasePositionAndOrientation(curr_gripper.body_id)
    x, y, z = curr_gripper.grab_start_pos
    curr_gripper.move_gripper(x, y, z + 0.3)
    for _ in range(50): p.stepSimulation(); time.sleep(1./240.)

    for _ in range(120): p.stepSimulation(); time.sleep(1./240.)

    current_obj.pos_grab_after, _ = p.getBasePositionAndOrientation(current_obj.body_id)
    curr_gripper.grab_end_pos, _ = p.getBasePositionAndOrientation(curr_gripper.body_id)

    result = curr_gripper.is_success()
    for _ in range(50): p.stepSimulation(); time.sleep(1./240.)

    return result, curr_gripper, current_obj


def trainloop(object_choice="cylinder",
              gripper_choice="pr2", gripper_urdf="pr2_gripper.urdf", csvfile="pr2_gripper_cylinder.csv"
              ):
    """Generate training data through simulation"""
    # ------------------- Setup ------------------- #
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0.2])
    p.setPhysicsEngineParameter(numSolverIterations=300, erp=0.3, contactERP=0.3)

    n = 10000
    target_samples = 150
    count1 = 0
    count0 = 0
    data_rows = []  # store only the rows we actually collect to avoid zero padding

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
    cols = ["x","y","z","qx","qy","qz","qw","Result"]
    df = pd.DataFrame(data_rows, columns=cols)
    print(df.head())
    df.to_csv(csvfile, index=False) 
    

def testphaseloop(classifier_type="logistic", object_choice="cylinder", 
                  gripper_choice="pr2", gripper_urdf="pr2_gripper.urdf", csvfile="pr2_gripper_cylinder.csv"):
    """Test the classifier on new grasp trials
    
    Args:
        classifier_type: "logistic" or "mlp" to choose classifier type
    """
    # ------------------- Setup ------------------- #
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
    cols = ["x","y","z","qx","qy","qz","qw","Result"]
    df = pd.DataFrame(data_rows, columns=cols)
    print(df.head())
    total_correct = sum(1 for entry in prediction_results if entry["correct"])
    print(f"\nClassifier matched {total_correct} / {len(prediction_results)} grasps.")
    

#-------------------------run the loops------------------------------------------#
# Comment out the trainloop() if you want to use the pre-made training data 

if __name__ == "__main__":
    # Run the loop that creates training data, Comment out the time.sleep() to fasten process.
    
    # trainloop(object_choice="cylinder",
    #          gripper_choice="pr2",
    #          gripper_urdf="pr2_gripper.urdf",
    #          csvfile="pr2_gripper_cylinder.csv")
    
    # Run this loop that trains the classifier model with a csv file ,
    # then it tests new 10 random grasps. Giving a success rate too.
    # Choose classifier_type="logistic" or classifier_type="mlp"
    testphaseloop(
        classifier_type="mlp",  # Change to "mlp" to use MLP classifier
        object_choice="cylinder", 
        gripper_choice="pr2", 
        gripper_urdf="pr2_gripper.urdf",
        csvfile="pr2_gripper_cylinder.csv"
    ) 


'''
----------------Different parameters you can have ------------------
If you would like to test the different gripper+object combinations you must
change the trainloop() and testphaseloop() parameters. 

CLASSIFIER TYPES:
- classifier_type="logistic" : Uses Logistic Regression with Polynomial Features
- classifier_type="mlp" : Uses Multi-Layer Perceptron Neural Network

GRIPPER + OBJECT COMBINATIONS:

1. Cylinder + Pr2 gripper
object_choice="cylinder", 
gripper_choice="pr2", gripper_urdf="pr2_gripper.urdf", csvfile="pr2_gripper_cylinder.csv"

2. Cube + Pr2 gripper
object_choice="cube", 
gripper_choice="pr2", gripper_urdf="pr2_gripper.urdf", csvfile="pr2_gripper_cube.csv"

3. Cube + Sdh gripper
object_choice="cube", 
gripper_choice="threefinger", gripper_urdf="./threeFingers/sdh/sdh.urdf", csvfile="sdh_gripper_cube.csv"

4. Cylinder + Sdh gripper
object_choice="cylinder", 
gripper_choice="threefinger", gripper_urdf="./threeFingers/sdh/sdh.urdf", csvfile="sdh_gripper_cylinder.csv"

'''
