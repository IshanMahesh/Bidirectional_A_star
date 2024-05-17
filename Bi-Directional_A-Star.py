#!/usr/bin/env python3

import numpy as np
import cv2
import math
from queue import PriorityQueue
import time
import matplotlib.pyplot as plt

class PathFinder:

    def __init__(self):
        #try:
            self.env_width = 600
            self.env_height = 300

            self.init_pos = (0, 0)
            self.final_pos = (599, 270)
            self.init_orientation = 0
            self.final_orientation = 180
            self.clearance_val = 2
            self.robot_size = 2
            self.weight_val = 1
            self.step_val = 10
            self.end_node = None
            self.goal_achieved = False
            self.create_environment()

            self.fwd_queue = PriorityQueue()
            self.fwd_queue.put((0, (self.init_pos, self.init_orientation)))
            self.fwd_edge_cost = {self.init_pos: 0}
            self.fwd_total_cost = {self.init_pos: 0}
            self.fwd_parent = {self.init_pos: self.init_pos}
            self.fwd_visited = [[[0 for _ in range(12)] for _ in range(self.env_width)] for _ in range(self.env_height)]
            self.fwd_visited_list = []
            self.fwd_angle_dict = {}

            self.bwd_queue = PriorityQueue()
            self.bwd_queue.put((0, (self.final_pos, self.final_orientation)))
            self.bwd_edge_cost = {self.final_pos: 0}
            self.bwd_total_cost = {self.final_pos: 0}
            self.bwd_parent = {self.final_pos: self.final_pos}
            self.bwd_visited = [[[0 for _ in range(12)] for _ in range(self.env_width)] for _ in range(self.env_height)]
            self.bwd_visited_list = []
            self.bwd_angle_dict = {}

            self.goal_threshold = 2 * (self.step_val + self.robot_size)

            self.find_path()
            self.visualize_path()

        #except:
            #print("Path Not found! Try changing input variables")

    def create_environment(self):
        self.env_data = np.zeros((self.env_height, self.env_width), dtype=int)
        self.env_visual = np.zeros((self.env_height, self.env_width), dtype=int)

        self.env_map = np.full((self.env_height, self.env_width, 3), 255, dtype=np.uint8)
        print("Create a Map with obstacles")

        for y in range(self.env_height):
            for x in range(self.env_width):
                if (y - 90) ** 2 + (x - 263) ** 2 <= 70 ** 2 or (y - 220) ** 2 + (x - 445) ** 2 <= 37.5 **2or (y - 242.5) ** 2 + (x - 112) ** 2 <= 40 ** 2:
                    self.env_data[y, x] = 1

        print("Map is being Created")

        for y in range(self.env_height):
            for x in range(self.env_width):
                if self.env_data[y, x] == 1:
                    self.env_map[self.env_height - y - 1, x] = (50, 220, 0)

        for y in range(self.env_height):
            for x in range(self.env_width):
                if (y - 90) ** 2 + (x - 263) ** 2 <= 70 ** 2 or (y - 220) ** 2 + (x - 445) ** 2 <= 37.5 **2or (y - 242.5) ** 2 + (x - 112) ** 2 <= 40 ** 2:
                    self.env_visual[y, x] = 1

        for y in range(self.env_height):
            for x in range(self.env_width):
                if self.env_visual[y, x] == 1:
                    self.env_map[self.env_height - y - 1, x] = (70, 70, 70)

    def mark_visited(self, node, direction):
        x, y, theta = node
        orientation = int(((360 + theta) % 360) / 30)
        if direction == "forward":
            self.fwd_visited[y][x][orientation] = True
        else:
            self.bwd_visited[y][x][orientation] = True

    def check_visited(self, node, direction):
        x, y, theta = node
        orientation = int(((360 + theta) % 360) / 30)
        if direction == "forward":
            for flag in self.fwd_visited[y][x]:
                if flag:
                    return True
        else:
            for flag in self.bwd_visited[y][x]:
                if flag:
                    return True
        return False

    def convert_coords(self, x, y):
        y_cv = (self.env_height - 1) - y
        x_cv = x
        return x_cv, y_cv

    def next_pos(self, node, angle_change):
        new_theta = (node[1] + angle_change) % 360
        updated_x = round(node[0][0] + self.step_val * math.cos(math.radians(new_theta)), 1)
        updated_y = round(node[0][1] + self.step_val * math.sin(math.radians(new_theta)), 1)
        return int(updated_x), int(updated_y), new_theta
    
    def get_neighbors(self, node):
        adjacent_nodes = []
        for degrees in [-90, -60, -30, 0, 30, 60, 90]:
            adj_node = self.next_pos(node, degrees)
            x, y = adj_node[:2]
            if 0 < x < self.env_width and 0 < y < self.env_height:
                if self.env_data[y][x] != 1:
                    adjacent_nodes.append(adj_node)
        return adjacent_nodes
    
    def visualize_path(self):
        start_x, start_y = self.convert_coords(self.init_pos[0], self.init_pos[1])
        goal_x, goal_y = self.convert_coords(self.final_pos[0], self.final_pos[1])
        node_count = 0

        for visited_node in self.combined_visited_list:
            node_count += 1

            xn, yn = self.convert_coords(visited_node[0], visited_node[1])
            self.env_map[yn, xn] = (150, 150, 150)
            angle = visited_node[2]
            step_size = self.step_val
            arrow_x = int(visited_node[0] + step_size * math.cos(np.pi + math.radians(angle)))
            arrow_y = int(visited_node[1] + step_size * math.sin(np.pi + math.radians(angle)))
            arrow_x, arrow_y = self.convert_coords(arrow_x, arrow_y)
            self.env_map = cv2.arrowedLine(self.env_map, (arrow_x, arrow_y), (xn, yn), (200, 200, 0), 1, tipLength=0.1)

            cv2.circle(self.env_map, (goal_x, goal_y), 6, (0, 0, 255), -1)
            cv2.circle(self.env_map, (start_x, start_y), 6, (0, 255, 0), -1)
            if node_count % 20 == 0:
                cv2.imshow("Map", self.env_map)
                cv2.waitKey(1)

        for node, angle in zip(self.robot_path, self.robot_angles):
            xn, yn = self.convert_coords(node[0], node[1])
            self.env_map[yn, xn] = (150, 150, 150)
            angle = angle
            step_size = self.step_val
            arrow_x = int(node[0] + step_size * math.cos(np.pi + math.radians(angle)))
            arrow_y = int(node[1] + step_size * math.sin(np.pi + math.radians(angle)))
            arrow_x, arrow_y = self.convert_coords(arrow_x, arrow_y)
            self.env_map = cv2.arrowedLine(self.env_map, (arrow_x, arrow_y), (xn, yn), (0, 0, 255), 2, tipLength=0.1)

            xn, yn = self.convert_coords(node[0], node[1])
            cv2.circle(self.env_map, (xn, yn), 1, (0, 255, 0), -1)
            cv2.imshow("Map", self.env_map)

            cv2.waitKey(1)
            time.sleep(0.05)

        cv2.imshow("Map", self.env_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def euclidean_dist(self, node1, node2):
        d = math.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2)
        return round(d, 1)

    def find_path(self):
        infinite = float('inf')

        while not self.fwd_queue.empty():
            fwd_total_cost, node = self.fwd_queue.get()
            node_coords = node[0]
            node_orientation = node[1]
            current_edge_cost = self.fwd_edge_cost.get(node_coords, infinite)

            for node_v in self.bwd_visited_list:
                nx, ny, nt = node_v
                if (nx, ny) == node_coords:
                    if abs(nt - node[1]) % 180 == 0:
                        self.end_node = node
                        print("Goal is Reached!")
                        self.goal_achieved = True
                        break

            if self.goal_achieved:
                break

            neighbors = self.get_neighbors(node)
            for adj_node in neighbors:
                adj_node_coords = adj_node[:2]
                added_edge_cost = self.euclidean_dist(node_coords, adj_node_coords)
                updated_edge_cost = self.fwd_edge_cost[node_coords] + added_edge_cost
                heuristic_cost = self.euclidean_dist(adj_node_coords, self.final_pos) * self.weight_val
                total_cost = heuristic_cost + updated_edge_cost

                if not self.check_visited(adj_node, "forward") or total_cost < self.fwd_total_cost.get(adj_node_coords, float('inf')):
                    self.fwd_edge_cost[adj_node_coords] = updated_edge_cost
                    self.fwd_total_cost[adj_node_coords] = total_cost
                    lowest_edge_cost = total_cost
                    orientation = ((360 + adj_node[2]) % 360) / 30
                    for x in range(int(node_coords[0]), int(adj_node[0])):
                        for y in range(int(node_coords[1]), int(adj_node[1])):
                            self.fwd_visited[y][x][int(orientation)] = 1
                    self.fwd_queue.put((lowest_edge_cost, (adj_node_coords, adj_node[2])))
                    self.fwd_parent[adj_node_coords] = node_coords
                    self.fwd_visited_list.append(adj_node)
                    self.mark_visited(adj_node, "forward")

            bwd_total_cost, node = self.bwd_queue.get()
            node_coords = node[0]
            node_orientation = node[1]
            current_edge_cost = self.bwd_edge_cost.get(node_coords, infinite)

            for node_v in self.fwd_visited_list:
                nx, ny, nt = node_v
                if (nx, ny) == node_coords:
                    if abs(nt - node[1]) % 180 == 0:
                        self.end_node = node
                        print("Goal is Reached!")
                        self.goal_achieved = True
                        break

            if self.goal_achieved==True:
                break

            neighbors = self.get_neighbors(node)
            for adj_node in neighbors:
                adj_node_coords = adj_node[:2]
                added_edge_cost = self.euclidean_dist(node_coords, adj_node_coords)
                updated_edge_cost = self.bwd_edge_cost[node_coords] + added_edge_cost
                heuristic_cost = self.euclidean_dist(adj_node_coords, self.init_pos) * self.weight_val
                total_cost = heuristic_cost + updated_edge_cost

                if not self.check_visited(adj_node, "backward") or total_cost < self.bwd_total_cost.get(adj_node_coords, float('inf')):
                    self.bwd_edge_cost[adj_node_coords] = updated_edge_cost
                    self.bwd_total_cost[adj_node_coords] = total_cost
                    lowest_edge_cost = total_cost
                    orientation = ((360 + adj_node[2]) % 360) / 30
                    for x in range(int(node_coords[0]), int(adj_node[0])):
                        for y in range(int(node_coords[1]), int(adj_node[1])):
                            self.bwd_visited[y][x][int(orientation)] = 1
                    self.bwd_queue.put((lowest_edge_cost, (adj_node_coords, adj_node[2])))
                    self.bwd_parent[adj_node_coords] = node_coords
                    self.bwd_visited_list.append(adj_node)
                    self.mark_visited(adj_node, "backward")

        for node in self.fwd_visited_list:
            x, y, angle = node
            self.fwd_angle_dict[(x, y)] = angle

        self.fwd_angle_dict[(self.init_pos[0], self.init_pos[1])] = self.init_orientation

        for node in self.bwd_visited_list:
            x, y, angle = node
            self.bwd_angle_dict[(x, y)] = angle

        self.bwd_angle_dict[(self.final_pos[0], self.final_pos[1])] = self.final_orientation

        self.robot_path = []
        self.robot_angles = []
        node = self.end_node
        node_coords = node[0]

        while node_coords != self.init_pos:
            self.robot_path.append(node_coords)
            node_coords = self.fwd_parent[node_coords]

        self.robot_path.append(self.init_pos)
        self.robot_path.reverse()

        for node in self.robot_path:
            self.robot_angles.append(self.fwd_angle_dict[node])

        self.reverse_path = []
        self.reverse_angles = []
        node = self.end_node
        node_coords = node[0]

        while node_coords != self.final_pos:
            self.reverse_path.append(node_coords)
            node_coords = self.bwd_parent[node_coords]

        self.reverse_path.append(self.final_pos)
        self.reverse_path.reverse()

        for node in self.reverse_path:
            self.reverse_angles.append(self.bwd_angle_dict[node])

        self.reverse_path.reverse()
        self.reverse_path.pop(0)
        self.reverse_angles.reverse()
        self.reverse_angles.pop()

        new_angles = []
        for angle in self.reverse_angles:
            new_angle = ((angle + 180) % 360)
            new_angles.append(new_angle)

        self.robot_path.extend(self.reverse_path)
        self.robot_angles.extend(new_angles)

        self.combined_visited_list = []

        fwd_length = len(self.fwd_visited_list)
        bwd_length = len(self.bwd_visited_list)

        max_length = max(fwd_length, bwd_length)

        for i in range(max_length):
            if i < fwd_length:
                self.combined_visited_list.append(self.fwd_visited_list[i])

            if i < bwd_length:
                self.combined_visited_list.append(self.bwd_visited_list[i])

if __name__ == "__main__":
    PathFinder()