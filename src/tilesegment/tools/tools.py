import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def get_labels_and_masks(labels_array):
    labels = []
    masks = []
    for lab in range(1, labels_array.max()):
        mask = np.transpose(np.where(labels_array == lab))
        labels.append(lab)
        masks.append(mask)
    return labels, masks

def get_diameter_distribution(labels_array):
    areas = []
    labels, masks = get_labels_and_masks(labels_array)
    for m, mask in enumerate(masks):
        area = len(mask)
        areas.append(area)
    
    return areas
    
# def remove_small_cells(labels_array, xyres):
#     labs_to_remove = []
#     areas = []
#     labels, masks = get_labels_and_masks(labels_array)
#     for m, mask in enumerate(masks):
#         area = len(mask)
#         areas.append(area)
#         if area < area_th:
#             labs_to_remove.append(labels[m])

#     np.mean(areas)
#     for lab in labs_to_remove:
import math
def distance_squared(point1, point2):
    # Calculate the squared Euclidean distance between two points
    return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2

def rotating_calipers(points):
    # Ensure the points are sorted in counterclockwise order
    # points.sort(key=lambda x: (x[0], x[1]))
    
    n = len(points)
    max_distance = 0
    i = 0
    j = 1
    
    while i < n:
        # Compute the current distance and update max_distance if necessary
        current_distance = distance_squared(points[i], points[j])
        max_distance = max(max_distance, current_distance)
        
        # Increment j to rotate the calipers
        j = (j + 1) % n
        
        # Check if we have completed a full rotation
        if j == i:
            i += 1
            j = (j + 1) % n
    
    return math.sqrt(max_distance)  # Return the square root of the maximum squared distance

from numba import njit

# Function to compute nearest neighbors for each node
def compute_nearest_neighbors(distance_matrix, k=5):
    num_nodes = distance_matrix.shape[0]
    nearest_neighbors = np.zeros((num_nodes, k)).astype('int32')
    nearest_neighbors_distances = np.zeros((num_nodes, k)).astype('float32')
    for i in range(num_nodes):
        # Sort distances and get indices of k nearest neighbors
        neighbors = np.argsort(distance_matrix[i])[:k]
        neighbors_dist = np.sort(distance_matrix[i])[:k]
        nearest_neighbors[i] = neighbors
        nearest_neighbors_distances[i] = neighbors_dist
    return nearest_neighbors, nearest_neighbors_distances

@njit
def compute2Ddistance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


@njit
def compute_distance_matrix(centers):
    ncells = centers.shape[0]
    distance_matrix = np.zeros((ncells, ncells))
    for i in range(ncells):
        centeri = centers[i]
        for j in range(i):
            centerj = centers[j]
            distance_matrix[i,j] = compute2Ddistance(centeri, centerj)
    
    distance_matrix+=np.transpose(distance_matrix)
    return distance_matrix

# distance_matrix = compute_distance_matrix(centers_graph)

def fill_diagonal_with(matrix, val):
    np.fill_diagonal(matrix, val, wrap=False)


# from https://gis.stackexchange.com/a/345439
def density_scatter_plot(x, y, **kwargs):
    """
    :param x: data positions on the x axis
    :param y: data positions on the y axis
    :return: matplotlib.collections.PathCollection object
    """
    # Kernel Density Estimate (KDE)
    values = np.vstack((x, y))
    kernel = gaussian_kde(values)
    kde = kernel.evaluate(values)

    # create array with colors for each data point
    norm = Normalize(vmin=kde.min(), vmax=kde.max())
    colors = cm.ScalarMappable(norm=norm, cmap='viridis').to_rgba(kde)

    # override original color argument
    kwargs['color'] = colors

    return plt.scatter(x, y, **kwargs)

import seaborn as sns
def kdeplot_log(x, y, **kwargs):
    """
    :param x: data positions on the x axis
    :param y: data positions on the y axis
    :return: matplotlib.collections.PathCollection object
    """
    return sns.kdeplot(np.log(x), np.log(y), **kwargs)


def increase_point_resolution(outline, min_outline_length):
    rounds = np.ceil(np.log2(min_outline_length / len(outline))).astype("int16")
    if rounds <= 0:
        newoutline_new = np.copy(outline)
    for r in range(rounds):
        if r == 0:
            pre_outline = np.copy(outline)
        else:
            pre_outline = np.copy(newoutline_new)
        newoutline_new = np.copy(pre_outline)
        i = 0
        while i < len(pre_outline) * 2 - 2:
            newpoint = np.array(
                [
                    np.rint((newoutline_new[i] + newoutline_new[i + 1]) / 2).astype(
                        "uint16"
                    )
                ]
            )
            newoutline_new = np.insert(newoutline_new, i + 1, newpoint, axis=0)
            i += 2
        newpoint = np.array(
            [np.rint((pre_outline[-1] + pre_outline[0]) / 2).astype("uint16")]
        )
        newoutline_new = np.insert(newoutline_new, 0, newpoint, axis=0)

    return newoutline_new
