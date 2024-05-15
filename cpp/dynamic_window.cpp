#include "sobol.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
enum shapes {
  RECTANGLE = 0,
  CIRCLE = 1,
};

double distance(double x1, double y1, double x2, double y2) {
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

double sign(double p1[2], double p2[2], double p3[2]) {
  return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
}

bool point_in_triangle(double point[2], double triangle[3][2]) {
  bool b1, b2, b3;
  b1 = sign(point, triangle[0], triangle[1]) < 0.0f;
  b2 = sign(point, triangle[1], triangle[2]) < 0.0f;
  b3 = sign(point, triangle[2], triangle[0]) < 0.0f;

  return ((b1 == b2) && (b2 == b3));
}

class DynamicWindowApproach {
public:
  DynamicWindowApproach(double max_speed, double max_steering,
                        double car_length, double car_width, int num_samples,
                        double dt, double duration, double run_freq,
                        double kspeed = 1, double kdistance = 1,
                        double kstability = 0.2, double kdirection = 1) {
    this->max_speed = max_speed;
    this->max_steering = max_steering;
    this->car_length = car_length;
    this->car_width = car_width;
    this->num_samples = num_samples;
    this->dt = dt;
    this->duration = duration;
    this->num_trajectory_points = (int)(duration / dt);
    this->run_freq = run_freq;
    this->kspeed = kspeed;
    this->kdistance = kdistance;
    this->kstability = kstability;
    this->kdirection = kdirection;
  }

  double *get_controls(double goal[2], double state[3], double *obstacles,
                       int obstacles_size) {

    double obs_speeds[obstacles_size * 2];
    std::fill_n(obs_speeds, obstacles_size * 2, 0);
    for (int i = 0; i < obstacles_size; i++) {
      if (obstacles[i * 5 + 3] && // obstacle is moving
          obs_history.find((int)obstacles[i * 5 + 4]) != obs_history.end()) {
        double obs_prev[2] = {obs_history[(int)obstacles[i * 5 + 4]][0],
                              obs_history[(int)obstacles[i * 5 + 4]][1]};
        obs_speeds[i * 2] = (obstacles[i * 5] - obs_prev[0]) / run_freq;
        obs_speeds[i * 2 + 1] = (obstacles[i * 5 + 1] - obs_prev[1]) / run_freq;
      }
    }

    double *best_controls = new double[2];
    best_controls[0] = 0;
    best_controls[1] = 0;
    double best_score = -INFINITY;
    double **motion_primitives = generate_motion_primitives();
    for (int i = 0; i < num_samples; i++) {
      double v = motion_primitives[i][0] * max_speed;
      double s = (motion_primitives[i][1] - 0.5) * max_steering * 2;

      double score = evaluate_score(v, s, goal, state, obstacles, obs_speeds,
                                    obstacles_size);

      if (score > best_score) {
        best_score = score;
        best_controls[0] = v;
        best_controls[1] = s;
      }
    }
    // keep obstacle history
    for (int i = 0; i < obstacles_size; i++) {
      obs_history[(int)obstacles[i * 5 + 4]][0] = obstacles[i * 5];
      obs_history[(int)obstacles[i * 5 + 4]][1] = obstacles[i * 5 + 1];
    }
    // clear motion primitives
    for (int i = 0; i < num_samples; i++) {
      delete[] motion_primitives[i];
    }
    delete[] motion_primitives;
    prev_v = best_controls[0];
    prev_s = best_controls[1];
    return best_controls;
  }

  double **create_trajectory(double state[3], double velocity,
                             double steering) {
    double **trajectory = new double *[num_trajectory_points];
    trajectory[0] = state;
    for (int i = 1; i < num_trajectory_points; i++) {
      double theta = trajectory[i - 1][2] + velocity * steering * dt / 5;
      double x = trajectory[i - 1][0] + velocity * cos(theta) * dt;
      double y = trajectory[i - 1][1] + velocity * sin(theta) * dt;
      trajectory[i] = new double[3]{x, y, theta};
    }
    return trajectory;
  }

private:
  double max_speed;
  double max_steering;
  double car_length;
  double car_width;
  int num_samples;
  double dt;
  double duration;
  int num_trajectory_points;
  double run_freq;
  double kspeed = 1;
  double kdistance = 1;
  double kstability = 0.1;
  double kdirection = 1;
  double prev_v = 0;
  double prev_s = 0;
  int seed = 0;
  std::map<int, double[2]> obs_history;

  double **generate_motion_primitives() {
    double **motion_primitives = new double *[num_samples];
    for (int i = 0; i < num_samples; i++) {
      float *motion_primitive = new float[2];
      i4_sobol(2, &seed, motion_primitive);
      double *motion_primitive_double = new double[2];
      motion_primitive_double[0] = motion_primitive[0];
      motion_primitive_double[1] = motion_primitive[1];
      motion_primitives[i] = motion_primitive_double;
      delete[] motion_primitive;
    }
    return motion_primitives;
  }

  bool check_collision(double state[3], double **trajectory, double *obstacles,
                       double *obs_speeds, int obstacles_size) {
    for (int i = 0; i < obstacles_size; i++) {
      double obstacle[3] = {obstacles[i * 5], obstacles[i * 5 + 1],
                            obstacles[i * 5 + 2]};
      for (int j = 0; j < num_trajectory_points; j++) {
        bool collision = false;
        double moved_obstacle[3] = {
            obstacle[0] + obs_speeds[i * 2] * dt * j,
            obstacle[1] + obs_speeds[i * 2 + 1] * dt * j, obstacle[2]};

        switch ((int)obstacles[i * 5 + 3]) {
        case CIRCLE:
          collision = check_circle_collision(trajectory[j], moved_obstacle);
          break;
        case RECTANGLE:
          collision = check_rectangle_collision(trajectory[j], obstacle);
          break;
        }
        if (collision) {
          return true;
        }
      }
    }
    return false;
  }
  bool check_circle_collision(double loc[3], double obstacle[3]) {
    double distance =
        sqrt(pow(loc[0] - obstacle[0], 2) + pow(loc[1] - obstacle[1], 2));
    return distance < obstacle[2] +
                          sqrt(pow(car_length, 2) + pow(car_width, 2)) +
                          0.03; // with error margin
  }
  bool check_rectangle_collision(double loc[3], double obstacle[3]) {
    double car_center[2] = {loc[0], loc[1]};

    double obstacle_size = obstacle[2] +
                           sqrt(pow(car_length, 2) + pow(car_width, 2)) +
                           0.1; // error margin

    double obstacle_upper_l[2] = {obstacle[0] - obstacle_size,
                                  obstacle[1] + obstacle_size};
    double obstacle_upper_r[2] = {obstacle[0] + obstacle_size,
                                  obstacle[1] + obstacle_size};
    double obstacle_lower_l[2] = {obstacle[0] - obstacle_size,
                                  obstacle[1] - obstacle_size};
    double obstacle_lower_r[2] = {obstacle[0] + obstacle_size,
                                  obstacle[1] - obstacle_size};

    double triangle1[3][2] = {{obstacle_upper_l[0], obstacle_upper_l[1]},
                              {obstacle_upper_r[0], obstacle_upper_r[1]},
                              {obstacle_lower_l[0], obstacle_lower_l[1]}};

    double triangle2[3][2] = {{obstacle_upper_r[0], obstacle_upper_r[1]},
                              {obstacle_lower_r[0], obstacle_lower_r[1]},
                              {obstacle_lower_l[0], obstacle_lower_l[1]}};

    return point_in_triangle(car_center, triangle1) ||
           point_in_triangle(car_center, triangle2);
  }

  double evaluate_score(double v, double s, double goal[2], double state[3],
                        double *obstacles, double *obs_speeds,
                        int obstacles_size) {
    double **trajectory = create_trajectory(state, v, s);
    if (check_collision(state, trajectory, obstacles, obs_speeds,
                        obstacles_size)) {
      return -INFINITY;
    }

    double speed_score = evaluate_speed_score(v);

    double distance_score =
        evaluate_distance_score(goal, trajectory[num_trajectory_points - 1]);
    double stability_score = evaluate_stability_score(v, s);
    double direction_score =
        evaluate_direction_score(goal, trajectory[num_trajectory_points - 1]);

    // clear trajectory
    for (int i = 1; i < num_trajectory_points; i++) {
      delete[] trajectory[i];
    }
    delete[] trajectory;

    return speed_score + distance_score + stability_score + direction_score;
  }

  double evaluate_speed_score(double v) { return kspeed * v; }

  double evaluate_distance_score(double goal[2], double car_position[2]) {
    return -kdistance *
           distance(goal[0], goal[1], car_position[0], car_position[1]);
  }
  double evaluate_stability_score(double v, double s) {
    return -kstability * (std::abs(s - prev_s));
  }

  double evaluate_direction_score(double goal[2], double car_position[3]) {
    double goal_angle =
        atan2(goal[1] - car_position[1], goal[0] - car_position[0]);
    double car_angle = car_position[2];
    double angle_diff = std::abs(goal_angle - car_angle);
    if (angle_diff > M_PI) {
      angle_diff = 2 * M_PI - angle_diff;
    }
    return -kdirection * angle_diff;
  }
};

extern "C" DynamicWindowApproach *DWA_new(double max_speed, double max_steering,
                                          double car_length, double car_width,
                                          int num_samples, double dt,
                                          double duration, double run_freq) {
  return new DynamicWindowApproach(max_speed, max_steering, car_length,
                                   car_width, num_samples, dt, duration,
                                   run_freq);
}
extern "C" double *DWA_get_controls(DynamicWindowApproach *dwa, double goal[2],
                                    double state[3], double *obstacles,
                                    int obstacles_size) {
  return dwa->get_controls(goal, state, obstacles, obstacles_size);
}

extern "C" double **DWA_create_trajectory(DynamicWindowApproach *dwa,
                                          double state[3], double velocity,
                                          double steering) {
  return dwa->create_trajectory(state, velocity, steering);
}

int main() {
  DynamicWindowApproach *dwa = DWA_new(3, 10, 0.16, 0.12, 64, 0.01, 1, 0.02);
  double goal[2] = {10, 10};
  double state[3] = {0, 0, 0};
  double *obstacles = new double[6];
  obstacles[0] = 5;
  obstacles[1] = 5;
  obstacles[2] = 1;
  obstacles[3] = 0;
  obstacles[4] = 6;
  obstacles[5] = 1;
  double *controls = DWA_get_controls(dwa, goal, state, obstacles, 2);

  std::cout << "Controls: " << controls[0] << ", " << controls[1] << std::endl;
  delete[] controls;
  delete[] obstacles;
  return 0;
}
