import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import bisect
import cv2
from statistics import median

def find_edges_window_based(intensities, window_size=8):
    kernel = np.ones(window_size) / window_size
    
    # Apply convolution with 'valid' mode to avoid zero padding issues
    smoothed_intensity = np.convolve(intensities, kernel, mode='valid')
    
    # Calculate the difference between the smoothed intensities
    intensity_diff = np.diff(smoothed_intensity, n=1)
    
    # Apply convolution to the differences to smooth them, still using 'valid' mode
    diff_smoothed = np.convolve(intensity_diff, kernel, mode='valid')
    
    # Normalize the differences
    normalized_diff = diff_smoothed / np.max(diff_smoothed, initial=1)
    
    # Detect edges where normalized differences are above a threshold (e.g., 90th percentile)
    threshold = np.percentile(np.abs(normalized_diff), 90)
    edges = np.where(np.abs(normalized_diff) > threshold)[0]

    # Adjust edges to align with the original array indices
    edges += window_size // 2 + 1  # Adjust for the window size and diff offset

    # # Plotting
    # plt.figure(figsize=(12, 6))
    
    # # Plot original intensities
    # plt.subplot(1, 2, 1)
    # plt.plot(intensities, label='Original Intensities', alpha=0.5)
    # plt.scatter(edges, [intensities[i] for i in edges if i < len(intensities)], color='red', label='Detected Edges')
    # plt.title('Original Intensities and Detected Edges')
    # plt.xlabel('Pixel Index')
    # plt.ylabel('Intensity Value')
    # plt.legend()

    # # Plot smoothed intensity and smoothed diffs
    # plt.subplot(1, 2, 2)
    # plt.plot(np.arange(window_size // 2, len(smoothed_intensity) + window_size // 2), smoothed_intensity, label='Smoothed Intensity')
    # plt.plot(np.arange(window_size + 1, len(diff_smoothed) + window_size + 1), diff_smoothed, label='Smoothed Diffs')
    # plt.scatter(edges, [smoothed_intensity[i - window_size // 2] for i in edges if i < len(smoothed_intensity) + window_size // 2], color='red', label='Edges on Smoothed')
    # plt.title('Smoothed Intensity and Differences')
    # plt.xlabel('Pixel Index')
    # plt.ylabel('Intensity Value')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()

    return edges

def extract_coordinate_from_edge_idx(edge_idx, image_array, nuclear_mask, angle_degrees, center, max_len):
    mask = np.zeros_like(image_array, dtype=np.uint8)
    angle_radians = np.deg2rad(angle_degrees)
    line_x_end = int(center[1] + max_len * np.cos(angle_radians))
    line_y_end = int(center[0] - max_len * np.sin(angle_radians))
    cv2.line(mask, center, (line_x_end, line_y_end), 255, 1)
    mask_coords = np.where(np.logical_and(mask,np.logical_not(nuclear_mask)))    
    edge = mask_coords[0][edge_idx],mask_coords[1][edge_idx]
    return edge


def normalize_edge_dict(edge_dict,window_size=7 , steps = 1080):
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
        

def plot_radial_intensity_with_window_edges(image_array, nuclear_mask, center, max_len, window_size=8, steps=1080):
    edge_dict = OrderedDict()
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
        edge_dict[angle_degrees] = edges 
        

    target_angle = list(edge_dict.keys())[90]  # Picking an example angle
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
    plt.title(f'Pixel Intensity along Radial Line at {target_angle}° from Nuclear Mask')
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
    for angle in edge_dict.keys():
        edge_idx = edge_dict[angle]
        for edge_idx in edge_dict[angle]:
            edge = extract_coordinate_from_edge_idx(int(edge_idx), image_array, nuclear_mask, angle, center, max_len)
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
    edge_idx = edge_dict[target_angle]
    for edge_idx in edge_dict[target_angle]:
        edge = extract_coordinate_from_edge_idx(int(edge_idx), image_array,nuclear_mask, target_angle, center, max_len)
        edges_x.append(edge[1])
        edges_y.append(edge[0])
    plt.scatter(edges_x,edges_y,color = 'purple', label = 'Edge')

    plt.tight_layout()
    plt.show()

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
plot_radial_intensity_with_window_edges(image_array, mask_array, center, distance)



#restrictions
#needs to form a closed loop, must start at theta in edge dict and return to theta
#lines from theta to -theta must satisfy min distance requirement, min width across
#restriction on the distance between adjacent angles
#Generate an edge dict with the maximum distance from the center to an edge of the cell
#generate an edge dict with what you expect average distance from the center to the edge of a cell to be
#traverse the combination of these edge dicts under the above restrictions
#start from known edges given by cv2
#if multiple paths take one with minimum sum of slope absolute values