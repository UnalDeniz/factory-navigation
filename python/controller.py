import ctypes

dwalib = ctypes.cdll.LoadLibrary("./build/libdwa.so")

dwalib.DWA_new.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
]
dwalib.DWA_new.restype = ctypes.c_void_p

dwalib.DWA_get_controls.argtypes = [
    ctypes.c_void_p,
    ctypes.c_double * 2,
    ctypes.c_double * 3,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]
dwalib.DWA_get_controls.restype = ctypes.POINTER(ctypes.c_double)


class DynamicWindowApproach(object):
    def __init__(
        self, max_speed, max_steering, car_length, car_width, num_samples, dt, duration
    ):
        self.obj = dwalib.DWA_new(
            max_speed, max_steering, car_length, car_width, num_samples, dt, duration
        )

    def get_controls(self, goal, state, obstacles):
        goal = (ctypes.c_double * 2)(*goal)
        state = (ctypes.c_double * 3)(*state)
        num_obstacles = len(obstacles)
        obstacles = (ctypes.c_double * (3 * num_obstacles))(
            *[item for array in obstacles for item in array]
        )
        return dwalib.DWA_get_controls(self.obj, goal, state, obstacles, num_obstacles)


dwa = DynamicWindowApproach(3.0, 10, 0.16, 0.12, 64, 0.01, 1.0)
goal = [10, 10]
state = [0, 0, 0]
obstacles = [[5, 5, 1], [0, 6, 1]]

controls = dwa.get_controls(goal, state, obstacles)
for i in range(3):
    print("Control:", controls[i])
