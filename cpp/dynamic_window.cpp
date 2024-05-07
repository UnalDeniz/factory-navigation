#include "sobol.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

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
                        double dt, double duration, double kspeed = 1,
                        double kdistance = 1, double kstability = 0.5) {
    this->max_speed = max_speed;
    this->max_steering = max_steering;
    this->car_length = car_length;
    this->car_width = car_width;
    this->num_samples = num_samples;
    this->dt = dt;
    this->duration = duration;
    this->num_trajectory_points = (int)(duration / dt);
    this->kspeed = kspeed;
    this->kdistance = kdistance;
    this->kstability = kstability;
  }

  double *get_controls(double goal[2], double state[3], double *obstacles,
                       int obstacles_size) {

    double *best_controls = new double[2];
    best_controls[0] = 0;
    best_controls[1] = 0;
    double best_score = -INFINITY;
    double **motion_primitives = generate_motion_primitives();
    for (int i = 0; i < num_samples; i++) {
      double v = motion_primitives[i][0] * max_speed;
      double s = (motion_primitives[i][1] - 0.5) * max_steering * 2;

      double score =
          evaluate_score(v, s, goal, state, obstacles, obstacles_size);

      if (score > best_score) {
        best_score = score;
        best_controls[0] = v;
        best_controls[1] = s;
      }
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

private:
  double max_speed;
  double max_steering;
  double car_length;
  double car_width;
  int num_samples;
  double dt;
  double duration;
  int num_trajectory_points;
  double kspeed = 1;
  double kdistance = 1;
  double kstability = 0.1;
  double prev_v = 0;
  double prev_s = 0;
  int seed = 0;

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

  bool check_collision(double state[3], double **trajectory, double *obstacles,
                       int obstacles_size) {
    for (int i = 0; i < obstacles_size; i++) {
      // if (distance(obstacles[i * 3], obstacles[i * 3 + 1], state[0],
      // state[1]) >
      //     obstacles[i * 3 + 2] * pow(2, 0.5) +
      //         sqrt(pow(car_length, 2) + pow(car_width, 2)) +
      //         max_speed * duration) {
      //   continue;
      // }
      for (int j = 0; j < num_trajectory_points; j++) {
        double car_center[2] = {trajectory[j][0], trajectory[j][1]};

        double obstacle_size = obstacles[i * 3 + 2] +
                               std::max(car_length, car_width) +
                               0.1; // error margin

        double obstacle_upper_l[2] = {obstacles[i * 3] - obstacle_size,
                                      obstacles[i * 3 + 1] + obstacle_size};
        double obstacle_upper_r[2] = {obstacles[i * 3] + obstacle_size,
                                      obstacles[i * 3 + 1] + obstacle_size};
        double obstacle_lower_l[2] = {obstacles[i * 3] - obstacle_size,
                                      obstacles[i * 3 + 1] - obstacle_size};
        double obstacle_lower_r[2] = {obstacles[i * 3] + obstacle_size,
                                      obstacles[i * 3 + 1] - obstacle_size};

        double triangle1[3][2] = {{obstacle_upper_l[0], obstacle_upper_l[1]},
                                  {obstacle_upper_r[0], obstacle_upper_r[1]},
                                  {obstacle_lower_l[0], obstacle_lower_l[1]}};

        double triangle2[3][2] = {{obstacle_upper_r[0], obstacle_upper_r[1]},
                                  {obstacle_lower_r[0], obstacle_lower_r[1]},
                                  {obstacle_lower_l[0], obstacle_lower_l[1]}};

        if (point_in_triangle(car_center, triangle1) ||
            point_in_triangle(car_center, triangle2)) {
          return true;
        }
      }
    }
    return false;
  }

  double evaluate_score(double v, double s, double goal[2], double state[3],
                        double *obstacles, int obstacles_size) {
    double **trajectory = create_trajectory(state, v, s);
    if (check_collision(state, trajectory, obstacles, obstacles_size)) {
      return -INFINITY;
    }

    double speed_score = evaluate_speed_score(v);
    double distance_score =
        evaluate_distance_score(goal, trajectory[num_trajectory_points - 1]);
    double stability_score = evaluate_stability_score(v, s);

    // clear trajectory
    for (int i = 1; i < num_trajectory_points; i++) {
      delete[] trajectory[i];
    }
    delete[] trajectory;

    return speed_score + distance_score + stability_score;
  }

  double evaluate_speed_score(double v) { return kspeed * v; }
  double evaluate_distance_score(double goal[2], double car_position[2]) {
    return -kdistance *
           distance(goal[0], goal[1], car_position[0], car_position[1]);
  }
  double evaluate_stability_score(double v, double s) {
    return -kstability * (std::abs(s - prev_s));
  }
};

extern "C" DynamicWindowApproach *DWA_new(double max_speed, double max_steering,
                                          double car_length, double car_width,
                                          int num_samples, double dt,
                                          double duration) {
  return new DynamicWindowApproach(max_speed, max_steering, car_length,
                                   car_width, num_samples, dt, duration);
}
extern "C" double *DWA_get_controls(DynamicWindowApproach *dwa, double goal[2],
                                    double state[3], double *obstacles,
                                    int obstacles_size) {
  return dwa->get_controls(goal, state, obstacles, obstacles_size);
}

int main() {
  DynamicWindowApproach *dwa = DWA_new(3, 10, 0.16, 0.12, 64, 0.01, 1);
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
