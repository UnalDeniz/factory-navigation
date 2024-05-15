import time
import random

import mujoco
import mujoco.viewer

import cmpe434_utils
import cmpe434_dungeon

import numpy as np
import scipy as sp

from controller import DynamicWindowApproach
from AStar.a_star import AStarPlanner

import visualizer

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
    # print("Distances:", distances)

    # Use boolean indexing to filter the coordinates based on the threshold distance
    filtered_coordinates = distances < threshold

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
    final_pos = random.choice(
        [key for key in tiles.keys() if tiles[key] == "." and key != start_pos]
    )

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
    for i, room in enumerate(rooms):
        obs_pos = random.choice(
            [tile for tile in room if tile != start_pos and tile != final_pos]
        )
        scene.worldbody.add(
            "geom",
            name="Z{}".format(i),
            type="cylinder",
            size=[0.2, 0.05],
            rgba=[0.8, 0.0, 0.1, 1],
            pos=[obs_pos[0] * 2, obs_pos[1] * 2, 0.08],
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

    m.opt.timestep = 0.002

    rooms = [m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("R")]
    obstacles = [m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("Z")]

    uniform_direction_dist = sp.stats.uniform_direction(2)
    obstacle_direction = [
        [x, y, 0] for x, y in uniform_direction_dist.rvs(len(obstacles))
    ]
    unused = np.zeros(1, dtype=np.int32)

    update_freq = 15

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:

        velocity = d.actuator("throttle_velocity")
        steering = d.actuator("steering")

        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        dynamic_window = DynamicWindowApproach(
            2.5, 10.0, 0.16, 0.12, 128, 0.1, 1, update_freq * m.opt.timestep
        )

        static_obs = np.array([])

        geom_id = 0

        for i in range(1, m.ngeom):
            obstacle = np.array(m.geom(i).pos[:2])
            obstacle = np.append(obstacle, max(m.geom(i).size[:2]))
            obstacle = np.append(obstacle, 0)
            obstacle = np.append(obstacle, i)
            if obstacle[2] != 1:
                continue
            static_obs = (
                np.vstack((static_obs, obstacle)) if static_obs.size else obstacle
            )

        astar = AStarPlanner(
            [p[0] for p in static_obs],
            [p[1] for p in static_obs],
            0.5,
            2**0.5 + (0.16**2 + 0.12**2) ** 0.5, # obj size + car size
        )
        path_x, path_y = astar.planning(start_coor[0], start_coor[1], goal[0], goal[1])
        path = np.vstack((path_x, path_y)).T
        # reverse the path
        path = path[::-1]
        geom_id = visualizer.visualize_path(
            path, viewer, geom_id, color=[1, 0, 0, 1], height=0.05
        )

        indices = np.arange(4, path.shape[0], 5)
        path = path[indices]
        path = np.vstack((path, goal))
        index = 0

        counter = 0

        while viewer.is_running() and time.time() - start < 3000:
            step_start = time.time()

            if not paused:

                # obstable update
                for i, x in enumerate(obstacles):
                    dx = obstacle_direction[i][0]
                    dy = obstacle_direction[i][1]

                    px = m.geom_pos[x][0]
                    py = m.geom_pos[x][1]
                    pz = 0.02

                    nearest_dist = mujoco.mj_ray(
                        m, d, [px, py, pz], obstacle_direction[i], None, 1, -1, unused
                    )

                    if nearest_dist >= 0 and nearest_dist < 0.4:
                        obstacle_direction[i][0] = -dy
                        obstacle_direction[i][1] = dx

                    m.geom_pos[x][0] = m.geom_pos[x][0] + dx * 0.001
                    m.geom_pos[x][1] = m.geom_pos[x][1] + dy * 0.001

                if counter % update_freq == 0:

                    car_pos = np.array(d.xpos[1].copy()[0:2])
                    close_static_obs = static_obs[
                        filter_coordinates(static_obs[:, :2], 2.5, car_pos)
                    ]

                    dynamic_obstacles = [
                        [m.geom_pos[x][0], m.geom_pos[x][1], m.geom_size[x][0], 1, x]
                        for x in obstacles
                    ]
                    dynamic_obstacles = np.array(dynamic_obstacles)

                    close_dynamic_obs = dynamic_obstacles[
                        filter_coordinates(dynamic_obstacles[:, :2], 2.5, car_pos)
                    ]
                    close_obs = np.vstack((close_static_obs, close_dynamic_obs))

                    car_state = np.append(car_pos, quaternion_to_euler(d.xquat[1])[2])
                    control = dynamic_window.get_controls(
                        path[index], car_state, close_obs
                    )

                    v = control[0]
                    s = control[1]

                    traj = dynamic_window.create_trajectory(car_state, v, s)
                    traj = [[traj[i][0], traj[i][1]] for i in range(1, 10)]
                    traj = np.array(traj)
                    visualizer.visualize_path(
                        traj, viewer, geom_id, color=[0, 0, 1, 1], height=0.05
                    )
                    old_geom_id = geom_id

                    velocity.ctrl = v
                    steering.ctrl = s

                counter = (counter + 1) % update_freq

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                if (
                    index != len(path) - 1
                    and np.linalg.norm(car_pos - path[index]) < 1.5
                ):
                    index += 1

                if index != len(path) - 1 and np.linalg.norm(
                    car_pos - path[index]
                ) > np.linalg.norm(car_pos - path[index + 1]):
                    index += 1

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
