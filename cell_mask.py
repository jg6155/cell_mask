import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import bisect
import cv2
from statistics import median
from scipy.ndimage import binary_dilation

def find_edges_window_based(intensities, window_size=5):
    smoothed_intensity = np.convolve(intensities,np.ones(window_size)/window_size, mode = 'same')
    smoothed_intensity = smoothed_intensity
    # Calculate the rolling average
    intensity_diff = np.diff(smoothed_intensity,n = 1)
    # Calculate the difference between the smoothed intensities
    diff_smoothed = np.abs(np.convolve(intensity_diff,np.ones(window_size)/window_size,mode = 'same'))
    
    # Normalize the differences
    normalized_diff = diff_smoothed / np.max(diff_smoothed, initial=1)
    edges = np.where(np.abs(normalized_diff) > np.percentile(np.abs(normalized_diff), 90))[0]
    if len(edges) == 0:
        edges = edges[-5:] if len(edges) > 5 else edges
    # plt.figure(figsize=(30, 6))
    # plt.subplot(1,3,1)
    # plt.plot(intensities, label='Original Intensities', alpha=0.5)
    # plt.plot(smoothed_intensity, label = 'Smoothed Intensity')
    # plt.plot(diff_smoothed, label = 'Smoothed diffs')
    # plt.scatter(edges, [intensities[edge] for edge in edges], color='green', label='Detected Edges')
    # plt.title('Effect of Convolution-Based Smoothing')
    # plt.xlabel('Pixel Index')
    # plt.ylabel('Intensity Value')
    # plt.legend()
    return edges

def extract_coordinate_from_edge_idx(edge_idx, image_array, angle_degrees, center, max_len):
    mask = np.zeros_like(image_array, dtype=np.uint8)
    angle_radians = np.deg2rad(angle_degrees)
    line_x_end = int(center[1] + max_len * np.cos(angle_radians))
    line_y_end = int(center[0] - max_len * np.sin(angle_radians))
    cv2.line(mask, center, (line_x_end, line_y_end), 255, 1)
    mask_coords = np.where(mask)
    if edge_idx >= len(mask_coords[0]):
        edge_idx = -1 
    
    edge = mask_coords[0][edge_idx],mask_coords[1][edge_idx]
    return edge


def normalize_edge_dict(edge_dict,window_size=2 , steps = 720):
    edge_keys = list(edge_dict.keys())
    edge_dict_copy = edge_dict.copy()
    for angle_degrees in np.linspace(0, 360, steps):
        adjusted_min_angle = (angle_degrees - window_size) % 360
        adjusted_max_angle = (angle_degrees + window_size) % 360
        
        if adjusted_min_angle < adjusted_max_angle:
            l, r = bisect.bisect_left(edge_keys, adjusted_min_angle), bisect.bisect_right(edge_keys, adjusted_max_angle)
            indices = edge_keys[l:r]
        else:  # Handle wrap around case
            l, r = bisect.bisect_left(edge_keys, 0), bisect.bisect_right(edge_keys, adjusted_max_angle)
            l2, r2 = bisect.bisect_left(edge_keys, adjusted_min_angle), bisect.bisect_right(edge_keys, 360)
            indices = edge_keys[l:r] + edge_keys[l2:r2]
        
        if indices:
            median_value = median([value for k in indices for value in edge_dict_copy[k]])
            edge_dict[angle_degrees] = median_value
        else:
            raise Exception("Missing gradient edges") 
        

def plot_radial_intensity_with_window_edges(image_array, nuclear_mask, center, max_len, window_size=8, steps=720):
    edge_dict = OrderedDict()
    for angle_degrees in np.linspace(0, 360, steps):
        # Creating a mask for each angle
        mask = np.zeros_like(image_array, dtype=np.uint8)
        angle_radians = np.deg2rad(angle_degrees)
        line_x_end = int(center[1] + max_len * np.cos(angle_radians))
        line_y_end = int(center[0] - max_len * np.sin(angle_radians))
        cv2.line(mask, center, (line_x_end, line_y_end), 255, 1)
        mask = mask > 0

        exclusion_mask = np.logical_not(nuclear_mask)
        combined_mask = mask*exclusion_mask

        line_intensities = image_array[combined_mask]
        edges = find_edges_window_based(line_intensities, window_size)
        edge_dict[angle_degrees] = edges
        

    target_angle = list(edge_dict.keys())[45]  # Picking an example angle
    mask = np.zeros_like(image_array, dtype=np.uint8)
    angle_radians = np.deg2rad(target_angle)
    line_x_end = int(center[1] + max_len * np.cos(angle_radians))
    line_y_end = int(center[0] - max_len * np.sin(angle_radians))
    cv2.line(mask, center, (line_x_end, line_y_end), 255, 1)
    mask = mask > 0
    line_intensities = image_array[mask]
    #normalize_edge_dict(edge_dict)
    # Plotting the pixel intensity profile for the target angle
    plt.figure(figsize=(30, 5))  # Setup figure size
    plt.subplot(1, 3, 1)  # Intensity profile subplot
    plt.plot(line_intensities, label='Pixel Intensity')
    plt.title(f'Pixel Intensity along Radial Line at {target_angle}° from Center')
    plt.scatter(edge_dict[target_angle], line_intensities[edge_dict[target_angle]], label = 'Edges', color = 'red')
    plt.xlabel('Distance from Center (pixels)')
    plt.ylabel('Pixel Intensity')
    plt.legend()

    # Plotting edges on the same line
    plt.subplot(1, 3, 2)  # Edges subplot
    edges_x = []
    edges_y = []
    plt.imshow(image_array, cmap = 'gray')
    plt.scatter(center[0],center[1], label = 'Center', color = 'green')
    for angle,edge_idx in edge_dict.items():
        edge = extract_coordinate_from_edge_idx(int(median(edge_idx)), image_array, angle, center, max_len)
        edges_x.append(edge[1])
        edges_y.append(edge[0])

    plt.scatter(edges_x, edges_y, color='red', label='Edges')
    plt.title('Detected Edges on the Line')
    plt.xlabel('Index in Intensity Array')
    plt.ylabel('Intensity Value')
    plt.legend()

    plt.subplot(1,3,3)
    cv2.line(image_array,center,(line_x_end, line_y_end), 255, 1)
    plt.imshow(image_array, cmap = 'gray') 
    plt.scatter(center[0],center[1], label = 'Center', color = 'green')
    edges_x = []
    edges_y = []
    for edge_idx in edge_dict[target_angle]:
        edge = extract_coordinate_from_edge_idx(int(edge_idx), image_array, target_angle, center, max_len)
        edges_x.append(edge[1])
        edges_y.append(edge[0])
    plt.scatter(edges_x,edges_y,color = 'purple', label = 'Edge')

    plt.tight_layout()
    plt.show()

    return edge_dict

# Example usage - replace with the actual image path and adjust parameters as needed
mask_path = '/Users/jorgegomez/Desktop/nuclear_mask.tif'
mask_array = np.array(Image.open(mask_path))

image_path = '/Users/jorgegomez/Desktop/denoised_image.tif'
image_array = np.array(Image.open(image_path))
center = (500, 500)  # OpenCV uses (x, y) format
distance = 220
angle_degrees = 0
angle_radians = np.deg2rad(angle_degrees)



plot_radial_intensity_with_window_edges(image_array,mask_array, center, distance)


