import time
import random

import mujoco
import mujoco.viewer

import cmpe434_utils
import cmpe434_dungeon

import numpy as np

from controller import DynamicWindowApproach
from AStar.a_star import AStarPlanner

final_pos = None
start_pos = None


def quaternion_to_euler(quaternion):
    # Convert quaternion to Euler angles using ZYX convention
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = (1 + 2 * (w * y - x * z)) ** 0.5
    cosp = (1 - 2 * (w * y - x * z)) ** 0.5
    pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def filter_coordinates(coordinates, threshold, reference_point):
    """Filter coordinates that are closer than the threshold distance to the reference point using NumPy."""
    # Convert the coordinates and the reference point to NumPy arrays
    # Calculate the distances between each coordinate and the reference point using vectorized operations
    distances = np.linalg.norm(coordinates - reference_point, axis=1)

    # Use boolean indexing to filter the coordinates based on the threshold distance
    filtered_coordinates = coordinates[distances < threshold]

    return filtered_coordinates


# Pressing SPACE key toggles the paused state.
# You can define other keys for other actions here.
def key_callback(keycode):
    if chr(keycode) == " ":
        global paused
        paused = not paused


paused = False  # Global variable to control the pause state.


def create_scenario():
    global final_pos
    global start_pos

    scene, scene_assets = cmpe434_utils.get_model("scenes/empty_floor.xml")

    tiles, rooms, connections = cmpe434_dungeon.generate(3, 2, 8)

    for index, r in enumerate(rooms):
        (xmin, ymin, xmax, ymax) = cmpe434_dungeon.find_room_corners(r)
        scene.worldbody.add(
            "geom",
            name="R{}".format(index),
            type="plane",
            size=[(xmax - xmin) + 1, (ymax - ymin) + 1, 0.1],
            rgba=[0.8, 0.6, 0.4, 1],
            pos=[(xmin + xmax), (ymin + ymax), 0],
        )

    for pos, tile in tiles.items():
        if tile == "#":
            scene.worldbody.add(
                "geom",
                type="box",
                size=[1, 1, 0.1],
                rgba=[0.8, 0.6, 0.4, 1],
                pos=[pos[0] * 2, pos[1] * 2, 0],
            )

    # Add the robot to the scene.
    robot, robot_assets = cmpe434_utils.get_model("models/mushr_car/model.xml")
    start_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])
    final_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])

    scene.worldbody.add(
        "site",
        name="start",
        type="box",
        size=[0.5, 0.5, 0.01],
        rgba=[0, 0, 1, 1],
        pos=[start_pos[0] * 2, start_pos[1] * 2, 0],
    )
    scene.worldbody.add(
        "site",
        name="finish",
        type="box",
        size=[0.5, 0.5, 0.01],
        rgba=[1, 0, 0, 1],
        pos=[final_pos[0] * 2, final_pos[1] * 2, 0],
    )

    start_yaw = random.randint(0, 359)
    robot.find("body", "buddy").set_attributes(
        pos=[start_pos[0] * 2, start_pos[1] * 2, 0.1], euler=[0, 0, start_yaw]
    )

    scene.include_copy(robot)

    # Combine all assets into a single dictionary.
    all_assets = {**scene_assets, **robot_assets}

    return scene, all_assets


def execute_scenario(scene, ASSETS=dict()):
    global final_pos
    global start_pos

    goal = np.array([final_pos[0] * 2, final_pos[1] * 2])
    start_coor = np.array([start_pos[0] * 2, start_pos[1] * 2])

    m = mujoco.MjModel.from_xml_string(scene.to_xml_string(), assets=all_assets)
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        
        velocity = d.actuator("throttle_velocity")
        steering = d.actuator("steering")

        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        dynamic_window = DynamicWindowApproach(3.0, 10.0, 0.16, 0.12, 64, 0.1, 1)

        obstacles = np.array([])

        geom_id = 0

        for i in range(1, m.ngeom):
            obstacle = np.array(m.geom(i).pos[:2])
            obstacle = np.append(obstacle, max(m.geom(i).size[:2]))
            if obstacle[2] != 1:
                continue
            obstacles = np.vstack((obstacles, obstacle)) if obstacles.size else obstacle

        astar = AStarPlanner(
            [p[0] for p in obstacles], [p[1] for p in obstacles], 0.5, 1.0
        )
        path_x, path_y = astar.planning(start_coor[0], start_coor[1], goal[0], goal[1])
        path = np.vstack((path_x, path_y)).T
        # reverse the path
        path = path[::-1]
        indices = np.arange(4, path.shape[0], 5)
        path = path[indices]
        path = np.vstack((path, goal))
        index = 0

        counter = 0

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_LINEBOX,
            size=np.array([0.01, 0.01, 1]),
            pos=np.append(path[index], 0.01),
            mat=[1, 0, 0, 0, 1, 0, 0, 0, 1],
            rgba=np.array([1, 1, 0, 1]),
        )
        geom_id += 1
        viewer.user_scn.ngeom = geom_id + 1
        viewer.sync()

        while viewer.is_running() and time.time() - start < 300:
            step_start = time.time()

            if not paused:

                if counter % 10 == 0:
                    car_pos = np.array(d.xpos[1].copy()[0:2])
                    close_obstacles = filter_coordinates(
                        obstacles, 3, np.append(car_pos, 1)
                    )

                    car_state = np.append(car_pos, quaternion_to_euler(d.xquat[1])[2])
                    control = dynamic_window.get_controls(
                        path[index], car_state, close_obstacles
                    )

                    v = control[0]
                    s = control[1]

                    old_geom_id = geom_id

                    # for point in traj:
                    #     print("Trajectory point:", point)
                    #     mujoco.mjv_initGeom(
                    #         viewer.user_scn.geoms[geom_id],
                    #         type=mujoco.mjtGeom.mjGEOM_LINEBOX,
                    #         size=np.array([0.15, 0.125, 1]),
                    #         pos=np.append(point[:2], 0.01),
                    #         # pos=np.array([xy[0], xy[1], 0.01]),
                    #         # mat=euler_angles_to_rotation_matrix(euler[2], 90, 0).flatten(),
                    #         mat=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                    #         rgba=np.array([0, 1, 1, 1]),
                    #     )
                    #     geom_id += 1
                    #     viewer.user_scn.ngeom = geom_id + 1
                    #     viewer.sync()

                    velocity.ctrl = v
                    steering.ctrl = s

                counter += 1

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                if (
                    index != len(path) - 1
                    and np.linalg.norm(car_pos - path[index]) < 1.5
                ):
                    index += 1
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geom_id],
                        type=mujoco.mjtGeom.mjGEOM_LINEBOX,
                        size=np.array([0.01, 0.01, 1]),
                        pos=np.append(path[index], 0.01),
                        mat=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                        rgba=np.array([1, 1, 0, 1]),
                    )
                    geom_id += 1
                    viewer.user_scn.ngeom = geom_id + 1

                if index != len(path) - 1 and np.linalg.norm(
                    car_pos - path[index]
                ) > np.linalg.norm(car_pos - path[index + 1]):
                    index += 1

                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geom_id],
                        type=mujoco.mjtGeom.mjGEOM_LINEBOX,
                        size=np.array([0.01, 0.01, 1]),
                        pos=np.append(path[index], 0.01),
                        mat=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                        rgba=np.array([1, 1, 0, 1]),
                    )
                    geom_id += 1
                    viewer.user_scn.ngeom = geom_id + 1

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                geom_id = old_geom_id

                if np.linalg.norm(car_pos - goal) < 0.5:
                    print("Goal reached!")
                    break

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return m, d


if __name__ == "__main__":
    scene, all_assets = create_scenario()
    execute_scenario(scene, all_assets)
