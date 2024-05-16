import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import cv2
import networkx as nx
import pickle

# Initialize an ordered dictionary to store edge information
edge_dict = OrderedDict()
keys = None
# Dictionary to map node IDs to their coordinates
node_to_coords = {}

def find_boundary_points(intensities, window_size=8):
    # Create a smoothing kernel
    kernel = np.ones(window_size) / window_size
    
    # Apply convolution with 'valid' mode to avoid zero padding issues
    smoothed_intensity = np.convolve(intensities, kernel, mode='valid')
    
    # Calculate the first derivative of the smoothed intensities
    first_derivative = np.diff(smoothed_intensity, n=1)
    first_derivative = np.convolve(first_derivative, kernel, mode = 'valid')
    
    # Calculate the 90th percentile of the first derivative
    percentile_value = np.percentile(first_derivative, 90)
    
    # Find the indices where the first derivative is greater than or equal to the 90th percentile
    percentile_indices = np.where(first_derivative >= percentile_value)[0]
    
    # Adjust the indices to align with the original array indices
    adjusted_indices = percentile_indices + window_size - 1  # Adjust for the window size and the diff offset

    return adjusted_indices #indices of boundary points along a ray

def weight_func(dist):
    return dist

def find_min_cost_path(start_angle, max_angle_jump=30, max_dist=20):
    print('finding path')
    angles = list(edge_dict.keys())
    n = len(angles)

    # Find the index of the specified start angle
    try:
        start_index = angles.index(start_angle)
    except ValueError:
        return []  # Start angle not found in the angle list

    # Construct the graph
    G = nx.DiGraph()
    print('constructing graph')
    for angle_idx, angle in enumerate(angles):
        print(f'node {angle}')
        for point_idx, (_, point) in enumerate(edge_dict[angle]):
            node_id = angle_idx * 100 + point_idx #if more than 100 potential boundary points on a ray, this must increase 
            G.add_node(node_id)
            node_to_coords[node_id] = point

            # Create edges between points within the allowable distance and angle jump
            for next_angle_idx in range(angle_idx + 1, angle_idx + max_angle_jump):
                next_angle_idx %= n
                for next_point_idx, (_, next_point) in enumerate(edge_dict[angles[next_angle_idx]]):
                    distance = np.linalg.norm(np.array(point) - np.array(next_point))
                    if distance <= max_dist:
                        next_node_id = next_angle_idx * 100 + next_point_idx
                        G.add_edge(node_id, next_node_id, weight=weight_func(distance)) ##
    print('graph constructed')
    # Finding the shortest path from the starting node to the ending node
    # Assuming a wrap around from the last angle to the first angle
    start_node = start_index * 100  # Assuming start from the first point of the start angle
    end_index = (start_index - 1) % n
    end_nodes = [end_index * 100 + i for i in range(len(edge_dict[angles[end_index]]))]
    print('looking for paths')
    min_path_cost = float('inf')
    best_path = []
    for end_node in end_nodes:
        try:
            path_length = nx.dijkstra_path_length(G, start_node, end_node)
            if path_length < min_path_cost:
                min_path_cost = path_length
                best_path = nx.dijkstra_path(G, start_node, end_node)
        except nx.NetworkXNoPath:
            print('no path found')
            continue  # No path exists
    print('path found')

    # # Plotting the subgraph
    # pos = nx.spring_layout(G)  # Positions for all nodes
    # path_edges = list(zip(best_path[:-1], best_path[1:]))
    # subgraph_nodes = set(best_path)
    # for node in best_path:
    #     subgraph_nodes.update(G.neighbors(node))  # Add neighbors of each node in the best path

    # # Create the subgraph
    # subgraph = G.subgraph(subgraph_nodes)

    # # Calculate positions for only the subgraph (optional, could use full graph's positions)
    # sub_pos = {node: pos[node] for node in subgraph_nodes}

    # # Plotting
    # plt.figure(figsize=(10, 8))
    # nx.draw(subgraph, sub_pos, node_size=10, with_labels=False, node_color='skyblue', edge_color='grey', linewidths=1, font_size=12)
    # nx.draw_networkx_nodes(subgraph, sub_pos, nodelist=best_path, node_color='red', node_size=30)
    # path_edges = list(zip(best_path[:-1], best_path[1:]))
    # nx.draw_networkx_edges(subgraph, sub_pos, edgelist=path_edges, edge_color='red', width=2)

    # plt.show()
    return best_path

def find_edges_on_img(image_array, nuclear_mask, center, max_len, window_size=8, steps=1080):
    print('finding edges')
    for angle_degrees in np.linspace(0, 360, steps):
        print(angle_degrees)
        # Creating a mask for each angle
        mask = np.zeros_like(image_array, dtype=np.uint8)
        angle_radians = np.deg2rad(angle_degrees)
        line_x_end = int(center[1] + max_len * np.cos(angle_radians))
        line_y_end = int(center[0] - max_len * np.sin(angle_radians))
        cv2.line(mask, center, (line_x_end, line_y_end), 255, 1)
        mask = mask > 0
        nuclear_exclusion = np.logical_not(nuclear_mask) 
        combined_mask = mask * nuclear_exclusion #exclude the nucleus

        line_intensities = image_array[combined_mask]
        edges = find_boundary_points(line_intensities, window_size)
        edge_coordinates = [(idx, np.column_stack(np.where(combined_mask))[idx]) for idx in edges]
        edge_dict[angle_degrees] = edge_coordinates
    print('found edges')
    return edge_dict

def merge_ordered_dicts(dict1, dict2):
    merged_dict = OrderedDict()
    for key in dict1.keys():
        merged_dict[key] = dict1[key] + dict2[key]
    return merged_dict

# Load mask images
mask_path = '/Users/jorgegomez/Desktop/nuclear_mask.tif'
mask2_path = '/Users/jorgegomez/Desktop/nuclear_mask2.tif'
mask_array = np.array(Image.open(mask_path), dtype=np.uint8)
mask2_array = np.array(Image.open(mask2_path), dtype=np.uint8)
center = (500, 500)
window_size = 8
distance = 170

# Combine masks using logical OR, convert to 32-bit signed integer
mask_array = np.logical_or(mask_array, mask2_array).astype(np.int32)

image_path = '/Users/jorgegomez/Desktop/test_img.tif'
image_array = np.array(Image.open(image_path))



with open('edge_dict.pkl', 'rb') as file:
    edge_dict = pickle.load(file)

# edge_dict1 = find_edges_on_img(image_array,mask_array,center, 230)
# edge_dict2 = find_edges_on_img(image_array,mask_array, center, 170)
# edge_dict = merge_ordered_dicts(edge_dict1,edge_dict2)

keys = list(edge_dict.keys())
start_angle = 270.2502316960148
x_coords, y_coords = [], []
for key in keys:
    points = edge_dict[key]
    for idx, (y, x) in points:
        x_coords.append(x)
        y_coords.append(y)

# Find the minimal cost path
path = find_min_cost_path(start_angle, 100, 50)

# Plot the edges and the path
plt.figure(figsize=(10, 10))
plt.scatter(x_coords, y_coords, s=4)

# Display the image
plt.imshow(image_array, cmap='gray')
plt.scatter(center[0], center[1], color='green', label='Center')

# Plot the path
y_coords, x_coords = zip(*[node_to_coords[node_id] for node_id in path])
plt.plot(x_coords, y_coords, marker='o', color='red', label='Path', markersize=2, lw=1)

plt.title('Minimal Cost Path')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.show()
