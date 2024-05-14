import numpy as np
import mujoco
import mujoco.viewer


def euler_angles_to_rotation_matrix(yaw, pitch, roll):
    # Convert Euler angles to radians
    # yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # Create individual rotation matrices
    R_yaw = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    R_pitch = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    R_roll = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    # Combine the rotation matrices (order: roll, pitch, yaw)
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))

    return R


def visualize_path(path, viewer, geom_id, color=[1, 0, 0, 1], height=0.05):
    for i in range(len(path) - 1):
        size = np.linalg.norm(path[i] - path[i + 1])
        yaw = np.arctan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=[0, 0, size],
            rgba=color,
            pos=np.array([path[i][0], path[i][1], height]),
            mat=euler_angles_to_rotation_matrix(yaw, 90, 0).flatten(),
        )
        geom_id += 1
        viewer.user_scn.ngeom = geom_id
        viewer.sync()

    return geom_id
