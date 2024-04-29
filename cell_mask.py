import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import cv2
import sys
import pickle
edge_dict = OrderedDict()
keys = None
sys.setrecursionlimit(50000)

def find_edges_window_based(intensities, window_size=8):
    kernel = np.ones(window_size) / window_size
    
    # Apply convolution with 'valid' mode to avoid zero padding issues
    smoothed_intensity = np.convolve(intensities, kernel, mode='valid')
    
    # Calculate the second derivative of the smoothed intensities
    second_derivative = np.diff(intensities, n=2)
    
    # Apply convolution to the second derivative to smooth it, still using 'valid' mode
    #second_derivative_smoothed = np.convolve(second_derivative, kernel, mode='valid')
    
    # Detect zero crossings
    signs = np.sign(second_derivative)
    zero_crossings = np.where(np.diff(signs) != 0)[0]

    # Adjust zero_crossings to align with the original array indices
    zero_crossings += window_size + 1  # Adjust for the window size and the diff offset

    return zero_crossings

def is_valid_step(last_coords, current_coords, max_dist):
    """Check if the distance between two points does not exceed max_dist."""
    return np.linalg.norm(np.array(last_coords) - np.array(current_coords)) <= max_dist

def is_valid_closed_path(path, max_dist):
    """Check if the last point to the first point distance is within max_dist."""
    return is_valid_step(path[-1][1], path[0][1], max_dist)

def calculate_slope_over_window(path, window_size):
    """Calculate the average slope over a window of angles in the path."""
    if len(path) < window_size:
        return 0  # Not enough points to calculate the desired window slope
    
    # Extract coordinates from the last `window_size` entries in the path
    _, coords = zip(*path[-window_size:])  
    x_coords, y_coords = zip(*coords)
    x_fit = np.polyfit(x_coords, y_coords, 1)  # Fit a linear model (poly of degree 1)
    y_fit = np.polyfit(y_coords,x_coords,1)
    slope = min(x_fit[0], y_fit[0]) # The slope of the linear fit
    return abs(slope)


#Use bellman-ford
#allow each node to have 30 edges consisting of the nodes in the adjacent degrees
#find mimimum cost to nodes that can return to the start
#close the loop
#maximum distance to outer edge
#expected distance to outer edges
#offset edge dictionaries by step_size
#add them together

def find_min_slope_path(current_idx, path, max_dist, cur_sum, max_angle_change, window_size):
    edge_dict
    


def plot_radial_intensity_with_window_edges(image_array, nuclear_mask, center, max_len, window_size=8, steps=1080):
    for angle_degrees in np.linspace(0, 360, steps):
        # Creating a mask for each angle
        mask = np.zeros_like(image_array, dtype=np.uint8)
        angle_radians = np.deg2rad(angle_degrees)
        line_x_end = int(center[1] + max_len * np.cos(angle_radians))
        line_y_end = int(center[0] - max_len * np.sin(angle_radians))
        cv2.line(mask, center, (line_x_end, line_y_end), 255, 1)
        mask = mask > 0
        nuclear_exclusion = np.logical_not(nuclear_mask)
        combined_mask = mask*nuclear_exclusion
        # plt.imshow(combined_mask,cmap = 'gray')
        # plt.show()
        line_intensities = image_array[combined_mask]
        edges = find_edges_window_based(line_intensities, window_size)
        edge_coordinates = [(idx, np.column_stack(np.where(combined_mask))[idx]) for idx in edges]
        edge_dict[angle_degrees] = edge_coordinates
    
        

    # target_angle = list(edge_dict.keys())[90]  # Picking an example angle
    # mask = np.zeros_like(image_array, dtype=np.uint8)
    # angle_radians = np.deg2rad(target_angle)
    # line_x_end = int(center[1] + max_len * np.cos(angle_radians))
    # line_y_end = int(center[0] - max_len * np.sin(angle_radians))
    # cv2.line(mask, center, (line_x_end, line_y_end), 255, 1)
    # mask = mask > 0
    # line_intensities = image_array[mask]
    # edge_indices, edge_coords = zip(*edge_dict[target_angle])
    # edge_indices = np.array(edge_indices)
    # #normalize_edge_dict(edge_dict)
    # # Plotting the pixel intensity profile for the target angle
    # plt.figure(figsize=(30, 5))  # Setup figure size
    # plt.subplot(1, 3, 1)  # Intensity profile subplot
    # plt.plot(line_intensities, label='Pixel Intensity')
    # plt.title(f'Pixel Intensity along Radial Line at {target_angle}° from Nuclear Mask')
    # plt.scatter(edge_indices, line_intensities[edge_indices], label = 'Edges', color = 'red')
    # plt.xlabel('Distance from Center (pixels)')
    # plt.ylabel('Pixel Intensity')
    # plt.legend()

    # # Plotting edges on the same line
    # plt.subplot(1, 3, 2)  # Edges subplot
    # edges_x = []
    # edges_y = []
    # plt.imshow(image_array, cmap = 'gray')
    # plt.scatter(center[0],center[1], label = 'Center', color = 'green')
    # for angle in edge_dict.keys():
    #     #edge_idx,edge = edge_dict[angle]
    #     for edge_idx,edge in edge_dict[angle]:
    #         edges_x.append(edge[1])
    #         edges_y.append(edge[0])

    # plt.scatter(edges_x, edges_y, color='red', label='Edges')
    # plt.title('Detected Edges on the Line')
    # plt.xlabel('Index in Intensity Array')
    # plt.ylabel('Intensity Value')
    # plt.legend()

    # plt.subplot(1,3,3)
    # cv2.line(image_array,center,(line_x_end, line_y_end), 255, 1)
    # plt.imshow(image_array, cmap = 'gray') 
    # plt.scatter(center[0],center[1], label = 'Center', color = 'green')
    # edges_x = []
    # edges_y = []
    # #edge_idx, edge = edge_dict[target_angle]
    # for edge_idx, edge in edge_dict[target_angle]:
    #     edges_x.append(edge[1])
    #     edges_y.append(edge[0])
    # plt.scatter(edges_x,edges_y,color = 'purple', label = 'Edge')

    # plt.tight_layout()
    # plt.show()

    return edge_dict

# Load mask images
mask_path = '/Users/jorgegomez/Desktop/nuclear_mask.tif'
mask2_path = '/Users/jorgegomez/Desktop/nuclear_mask2.tif'
mask_array = np.array(Image.open(mask_path), dtype=np.uint8)
mask2_array = np.array(Image.open(mask2_path), dtype=np.uint8)
center = (500,500)
window_size = 8
distance = 170

# Combine masks using logical OR, convert to 32-bit signed integer
mask_array = np.logical_or(mask_array, mask2_array).astype(np.int32)

image_path = '/Users/jorgegomez/Desktop/test_img.tif'
image_array = np.array(Image.open(image_path))
#plot_radial_intensity_with_window_edges(image_array, mask_array, center, distance)

with open('edge_dict.pkl', 'rb') as file:
    edge_dict = pickle.load(file)
# with open('edge_dict2.pkl','rb') as file:
#     edge_dict2 = pickle.load(file)
keys = list(edge_dict.keys())
# for key in keys:
#     edge_dict[key].extend(edge_dict2[key])




keys = list(edge_dict.keys()) 
start = keys.index(270.2502316960148)
x_coords, y_coords = [], []
for key in keys:
    points = edge_dict[key]
    print(edge_dict[key])
    for idx, (y,x) in points:
        x_coords.append(x)
        y_coords.append(y)

path,_ = find_min_slope_path(start,[(start, edge_dict[270.2502316960148][0][1])],200,0,2,5)
print(path)
#print(edge_dict[270.2502316960148])
#path, min_sum = find_min_slope_path(start,200,1,2,5)
#print(path,min_sum)
# print(min_sum)



plt.figure(figsize=(10, 10))
plt.scatter(x_coords,y_coords)


# Display the image
plt.imshow(image_array, cmap='gray')
plt.scatter(center[0], center[1], color='green', label='Center')
plt.scatter(edge_dict[270.2502316960148][0][1][1],edge_dict[270.2502316960148][0][1][0], label = 'Start')

# Plot the path
y_coords, x_coords = zip(*[coords for _, coords in path])
plt.plot(x_coords, y_coords, marker='o', color='red', label='Path')

plt.title('Minimal Slope Path on the Image')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.show()



# needs to form a closed loop, must start at theta in edge dict and return to theta
# lines from theta to -theta must satisfy min distance requirement, min width across
# restriction on the distance between adjacent angles (smoothness) 
# Generate an edge dict with the maximum distance from the center to an edge of the cell
# generate an edge dict with what you expect average distance from the center to the edge of a cell to be
# traverse the combination of these edge dicts under the above restrictions
# start from known edges given by cv2
# if multiple paths take one with minimum sum of slope absolute values


