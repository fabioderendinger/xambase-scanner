import numpy as np
from numpy import linalg as LA
import cv2
import math
import timeit

################################################################################################################################################################################
################################################################################################################################################################################
#                                                                   Function Definition
################################################################################################################################################################################
################################################################################################################################################################################


def resize_image(img, target_size):
    "Resize the image so that side length the longest side is equal to target_size. \
    Generally, the img size is significantly reduced and the Line Segment Detection (LSD) algorithm runs on the reduced image for better performance and results"
    # img: numpy matrix
    # target_size: positive integer

    s = target_size
    i = 0
    height, width = img.shape[:2]
    wh_max = max(height, width)
    if wh_max < 2 * target_size:
        resize_factor = s / wh_max
        return cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    else:
        while s <= wh_max:
            s = s * 2
            i += 1
        img = cv2.resize(img, None, fx=s / 2 / wh_max, fy=s / 2 / wh_max, interpolation=cv2.INTER_AREA)
        while i > 1:
            img = cv2.pyrDown(img)
            i -= 1
        return img

def line_segment_detector(img):
    "Detect the line segments in the input image using the LSD algorithm with the custom parameters defined in the section PARAMETERS"
    # img: numpy matrix
    return cv2.createLineSegmentDetector(_refine, _scale, _sigma_scale, _quant, _ang_th, _log_eps, _density_th, _n_bins).detect(img)[0]




################################################################################################################################################################################
# Section: Line Dictionary
# Purpose: The Line Dictionary stores the line segments returned by the LSD algorithm in a Python Dictionary together with key charactaristics of each line
################################################################################################################################################################################

def build_line_dictionary(lines, img_size):
    "Stores the line segments returned by the LSD algorithm in a Python Dictionary together with key charactaristics of each line (e.g. length, angle wrt to x axis, location, etc.)"
    # lines: numpy matrix representing the line segments returned by the LSD algorithm
    # img_size: 2x1 tuple

    dict = {}
    i = 0
    if np.any(lines):
        for line in lines:
            for e in line:
                array = leftpoint(e)
                vector, length, alpha = length_and_angle(array)
                location = find_location(array, alpha, img_size)
                if location and location[2]>0.5:
                    ulen = utility_length(length, location[1])
                    total_utility = total_utility_line(ulen,location[2])
                    dict[i] = {"line_seg": array,
                               "angle": alpha,
                               "length": [length, ulen],
                               "location": location,
                               "utility": total_utility,
                               "type": 0,
                               "status": 1}
                    i += 1
    return dict




################################################################################################################################################################################
# Section: Line Dictionary
# Subsection: Key characteristics of line segments
# Purpose: Calculate key characteristics such as length, angle, location, etc. for each line segment
################################################################################################################################################################################

def leftpoint(line):
    "The input line is a 4x1 numpy array representing the endpoints of a line segment. This functions re-arranges the end points of a line segment so that the lefter endpoint is first in the array"
    # line: 4x1 numpay array representing the endpoints
    if line[0] < line[2]:
        return line
    else:
        return np.array([line[2], line[3], line[0], line[1]])


def length_and_angle(line):
    "Returns the vector spanned by the two endpoints in line, the length of the line segment and the angle between the line and the positive x axis in"
    # line: 4x1 numpay array representing the endpoints
    vector = line[2:4] - line[0:2]
    vector_len = LA.norm(vector)
    x_unit = np.array([1, 0])
    if vector[1] < 0:
        return vector, vector_len, math.degrees(np.arccos(np.dot(vector, x_unit) / vector_len))
    return vector, vector_len, -math.degrees(np.arccos(np.dot(vector, x_unit) / vector_len))


def find_location(line, angle, img_size):
    "Returns the mid-point of a line, the location of the mid-point and the utility of the location. The mid-point can be in the top, right, bottom or left of an image"
    # line: 4x1 numpay array representing the endpoints
    # angle: float in [90,-90)
    # img-size: 2x1 tuple
    mid = (line[2:4] + line[0:2]) / 2
    half_height = img_size[0] / 2
    half_width = img_size[1] / 2
    if mid[0] < half_width and (angle > 70 or angle < -80):
        return [mid, "left", utility_left(mid, img_size , region)]
    if mid[0] >= half_width and (angle > 80 or angle < -70):
        return [mid, "right", utility_right(mid, img_size , region)]
    if mid[1] < half_height and abs(angle) < 10:
        return [mid, "top", utility_top(mid, img_size , region)]
    if mid[1] >= half_height and abs(angle) < 10:
        return [mid, "bottom", utility_bottom(mid, img_size , region)]
    return None


def avg_color(img, line, line_vector, length, location, width):
    "Calculates the average color intensity of the pixel values around a line segment. For instance, for a line located in the top half of \
    the image the function will return the average color intensity accross all pixels lying in a rectangle below the line of predefined width"
    # img: numpy matrix containing the gray-values (color intensity)
    # line: 4x1 numpay array representing the endpoints
    # line_vector: 2x1 numpy array (vector spanned by the endpoints of the line)
    # length: float (length of the line)
    # location: string (location of the line; can be {top, right, bottom, left])
    # width: float (width of the rectangle from which the color intensities are taken to calculate the average)
    if location == "top":
        color_values = []
        line_vector = line_vector / length
        normal_vec = np.array([-line_vector[1]/line_vector[0],1])
        normal_vec = normal_vec / LA.norm(normal_vec)
        start = line[0:2]
        for i in range(1,int(length)-1):
            for j in range(2,int(width)+2):
                x,y = np.around(start + i * line_vector + j*normal_vec)
                #img[int(y), int(x)]=255
                color_values.append(img[int(y),int(x)])
        if color_values:
            return sum(color_values) / len(color_values)
    if location == "bottom":
        color_values = []
        line_vector = line_vector / length
        normal_vec = np.array([-line_vector[1]/line_vector[0],-1])
        normal_vec = normal_vec / LA.norm(normal_vec)
        start = line[0:2]
        for i in range(1,int(length)-1):
            for j in range(2,int(width)+2):
                x,y = np.around(start + i * line_vector + j*normal_vec)
                #img[int(y), int(x)] = 255
                color_values.append(img[int(y),int(x)])
        if color_values:
            return sum(color_values) / len(color_values)
    if location == "left":
        color_values = []
        line_vector = line_vector / length
        normal_vec = np.array([1,-line_vector[0]/line_vector[1]])
        normal_vec = normal_vec / LA.norm(normal_vec)
        start = line[0:2]
        for i in range(1,int(length)-1):
            for j in range(2,int(width)+2):
                x,y = np.around(start + i * line_vector + j * normal_vec)
                #img[int(y), int(x)] = 255
                color_values.append(img[int(y),int(x)])
        if color_values:
            return sum(color_values) / len(color_values)
    if location == "right":
        color_values = []
        line_vector = line_vector / length
        normal_vec = np.array([-1,-line_vector[0]/line_vector[1]])
        normal_vec = normal_vec / LA.norm(normal_vec)
        start = line[0:2]
        for i in range(1,int(length)-1):
            for j in range(2,int(width)+2):
                x,y = np.around(start + i * line_vector + j*normal_vec)
                #img[int(y), int(x)] = 255
                color_values.append(img[int(y),int(x)])
        if color_values:
            return sum(color_values) / len(color_values)
    else:
        return None




################################################################################################################################################################################
# Section: Line Dictionary
# Subsection: Utility of line segment
# Purpose: Calculate the utility of a line segment based on its location in the image and its length. The utility is a measure used to identify the best document edge candidates among all lines
################################################################################################################################################################################

def utility_length(length, location):
    "Calculates the utility of a line based on its length and location"
    # length: integer
    # location: {top, right, bottom, left}
    if location == "top" or location == "bottom":
        return length / region["max_paper"][1]
    else:
        return length / region["max_paper"][0]


def main_region(img_size, parameters):
    "Defines the region in the image which likely contains the edges of the scanned document"
    # parameters: 3x1 list where parameters[0]: paper_ratio; parameters[1]: min_size_ratio; parameters[2]: boarder_margin
    # img_size: 2x1 tuple

    # if img_size[0]/img_size[1] > parameters[0]:
    max_paper_height = img_size[1] * parameters[0]
    y_max = (img_size[0] - max_paper_height) / 2
    y_min = (img_size[0] - max_paper_height * parameters[1]) / 2
    y_max_con = min(parameters[2] * y_max, y_max * math.sqrt(y_max / max_paper_height))
    y_min_con = min(img_size[0] / 2, y_min + (y_max - y_max_con))
    x_max = img_size[1] * (1 - parameters[1]) / 2
    region = {"x_bounds": [0, x_max, img_size[1] - x_max, img_size[1]],
            "y_bounds": [y_max_con, y_min_con, img_size[0] - y_min_con, img_size[0] - y_max_con],
            "mid": [img_size[0] / 2, img_size[1] / 2],
            "dist": [img_size[0] / 2 - y_min_con, img_size[1] / 2 - x_max],
            "max_paper": [max_paper_height,img_size[1]]}
    return region


def utility_top(mid_point, img_size, region):
    "Calculates the location utility for a line segment which's mid-point lies in the top half of the image. The location utility is a measure for how well the line segment lies in the main region where the edges of the scanned document are expected"
    # mid_point: 2x1 numpy array
    # img_size: 2x1 tuple
    # region: Dictionary returned by main_region() with the key characteristics defining the region
    if mid_point[1] >= region["y_bounds"][0] and mid_point[1] <= region["y_bounds"][1]:
        return 1
    elif mid_point[1] < region["y_bounds"][0]:
        print("top1 ", (mid_point[1] / region["y_bounds"][0]) ** 2)
        return (mid_point[1] / region["y_bounds"][0]) ** 2
    else:
        print("top2 ", ((region["mid"][0] - mid_point[1]) / region["dist"][0]) ** 2)
        return ((region["mid"][0] - mid_point[1]) / region["dist"][0]) ** 2


def utility_bottom(mid_point, img_size, region):
    "Calculates the location utility for a line segment which's mid-point lies in the bottom half of the image. The location utility is a measure for how well the line segment lies in the main region where the edges of the scanned document are expected"
    # mid_point: 2x1 numpy array
    # img_size: 2x1 tuple
    # region: Dictionary returned by main_region() with the key characteristics defining the region
    if mid_point[1] >= region["y_bounds"][2] and mid_point[1] <= region["y_bounds"][3]:
        return 1
    elif mid_point[1] > region["y_bounds"][3]:
        print("bottom1 ", ((img_size[0] - mid_point[1]) / region["y_bounds"][0]) ** 2)
        return ((img_size[0] - mid_point[1]) / region["y_bounds"][0]) ** 2
    else:
        print("bottom2 ", ((mid_point[1] - region["mid"][0]) / region["dist"][0]) ** 2)
        return ((mid_point[1] - region["mid"][0]) / region["dist"][0]) ** 2


def utility_left(mid_point, img_size, region):
    "Calculates the location utility for a line segment which's mid-point lies in the left half of the image. The location utility is a measure for how well the line segment lies in the main region where the edges of the scanned document are expected"
    # mid_point: 2x1 numpy array
    # img_size: 2x1 tuple
    # region: Dictionary returned by main_region() with the key characteristics defining the region
    if mid_point[0] >= region["x_bounds"][0] and mid_point[0] <= region["x_bounds"][1]:
        return 1
    elif mid_point[0] > region["x_bounds"][1]:
        print("left1 ", ((region["mid"][1] - mid_point[0]) / region["dist"][1]) ** 2)
        return ((region["mid"][1] - mid_point[0]) / region["dist"][1]) ** 2
    else:
        print("left2 ", (mid_point[0] / region["x_bounds"][0]) ** 2)
        return (mid_point[0] / region["x_bounds"][0]) ** 2


def utility_right(mid_point, img_size, region):
    "Calculates the location utility for a line segment which's mid-point lies in the right half of the image. The location utility is a measure for how well the line segment lies in the main region where the edges of the scanned document are expected"
    # mid_point: 2x1 numpy array
    # img_size: 2x1 tuple
    # region: Dictionary returned by main_region() with the key characteristics defining the region
    if mid_point[0] >= region["x_bounds"][2] and mid_point[0] <= region["x_bounds"][3]:
        return 1
    elif mid_point[0] < region["x_bounds"][2]:
        print("right1 ", ((mid_point[0] - region["mid"][1]) / region["dist"][1]) ** 2)
        return ((mid_point[0] - region["mid"][1]) / region["dist"][1]) ** 2
    else:
        print("right2 ", ((img_size[1] - mid_point[0]) / region["x_bounds"][0]) ** 2)
        return ((img_size[1] - mid_point[0]) / region["x_bounds"][0]) ** 2


def total_utility_line(ulen, uloc):
    "Calculates the total utility of a line segment as the weighted sum of the length utiltiy and location utility"
    # ulen: float
    # uloc: float
    return 0.5*ulen + 0.5*uloc




################################################################################################################################################################################
# Section: Connected Components
# Purpose: In an image, the edges of a document are not always bounded by straight lines and as a result a edge can be made up of multiple shorter line segments.
# The purpose of below functions is to identify line segments which lie close to each others and would approximately represent a straight line when connected.
################################################################################################################################################################################

def are_connected(line_a, angle_a, line_b, angle_b, radius, rho):
    "Identifies if two lines a and b are connected. We say that two lines a and b are connected if the distance between two of their endpoints is smaller than some radius r and the slope delta of the lines (defined by the angle) does not exceed some angle rho"
    # line_a: 4x1 numpay array representing the endpoints
    # line_b: 4x1 numpay array representing the endpoints
    # angle_a: # angle: float in [90,-90) representing the angle between line_a and the positive x axis
    # angle_b: # angle: float in [90,-90) representing the angle between line_b and the positive x axis
    # radius: float (maximum distance between the two lines)
    # rho: float (maximum angle delta between the two lines)
    if (abs(angle_a - angle_b) > rho) and (abs(abs(angle_a - angle_b) - math.degrees(math.pi)) > rho):
        return False

    len_a = LA.norm(line_a[2:4] - line_a[0:2])
    len_b = LA.norm(line_b[2:4] - line_b[0:2])
    max_ab = max(len_a, len_b)

    dist_ab1 = LA.norm(line_a[2:4] - line_b[0:2])
    dist_ab2 = LA.norm(line_a[0:2] - line_b[2:4])
    dist_ab3 = LA.norm(line_a[0:2] - line_b[0:2])
    dist_ab4 = LA.norm(line_a[2:4] - line_b[2:4])

    if dist_ab1 < radius and dist_ab2 > max_ab:
        return True
    if dist_ab2 < radius and dist_ab1 > max_ab:
        return True
    if dist_ab3 < radius and dist_ab4 > max_ab:
        return True
    if dist_ab4 < radius and dist_ab3 > max_ab:
        return True

    return False


def adjacency_matrix(lines, radius, rho):
    "Returns a nxn numpy square matrix where n is the number of lines returned by the LSD algorithm. The elements of the matrix indicate whether a pair of lines is connected. \
    When a line A and a line B are connected then the matrix value is matrix(A,B) = matrix(B,A) = 1. If they are not connected the value is 0"
    # lines: numpy matrix representing the line segments returned by the LSD algorithm
    # radius: float (maximum distance between the two lines)
    # rho: float (maximum angle delta between the two lines)
    n = len(lines)
    adjMatrix = np.zeros([n, n])

    for i in range(1, n):
        start = timeit.default_timer()
        for j in range(i, n):
            value = are_connected(lines[i - 1]["line_seg"], lines[i - 1]["angle"], lines[j]["line_seg"],
                                  lines[j]["angle"], radius, rho)
            adjMatrix[i - 1, j] = value
            adjMatrix[j, i - 1] = value
        end = timeit.default_timer()
        print('cycle: ', i, ' | time', end - start)
    return adjMatrix


def connected_components(adjMatrix, line_dict, img_size):
    "Given the adjacency matrix of conntected lines, this function identifies the connected components. In our case, a connected component is a set of 2 or more lines which are connected.\
    The connected components are identified via a depth-first search where we start with some root line, identify the lines connected to the root, continue identifying the lines connected to the line connected to the root, and so on. \
    To find all the connected components, we loop through the lines, starting a new depth-first search whenever the loop reaches a root line that has not already been included in a previously found connected component.\
    \
    The line resulting from a connected component is added to the dictionary of lines together with the key characteristics of this line"
    # adjMatrix: numpy square matrix
    # line_dict: Python dictionary containing the original line segments returned by the LSD algorithm
    # img_size: 2x1 tuple

    n = len(adjMatrix)
    visited = np.zeros(n)
    j = n
    for i in range(0, n):
        if visited[i] == False:
            new_component = [i]
            stack = [i]
            while stack:
                line_index = stack.pop()
                if visited[line_index] == False:
                    neighbours = get_all_connected_lines(adjMatrix, line_index)
                    stack = stack + neighbours
                    update_component(neighbours, new_component)
                    visited[line_index] = True
            if len(new_component) > 1:
                line = line_from_connected_lines(line_dict, new_component)
                vector, length, alpha = length_and_angle(line)
                location = find_location(line, alpha, img_size)
                ulen = utility_length(length, location[1])
                total_utility = total_utility_line(ulen,location[2])
                line_dict[j] = {"component": new_component,
                                "line_seg": line,
                                "angle": alpha,
                                "length": [length, ulen],
                                "location": location,
                                "utility": total_utility,
                                "type": 1,
                                "status": 1}
                inactivate_lines(line_dict, new_component)
                j += 1



def get_all_connected_lines(adjMatrix, line_index):
    "Returns the dictionary indices of lines connected to a line with index line_index via the adjacency matrix"
    # adjMatrix: numpy square matrix
    # line_index: integer (index of the line in the line dictionary)
    neighbours = []
    i = 0
    for e in adjMatrix[line_index, :]:
        if e:
            neighbours.append(i)
        i += 1
    return neighbours


def update_component(neighbours, component):
    "Adds the elements of NEIGHBOURS which are not already contained in COMPONENT to COMPONENT"
    # neighbours: list (list of indices)
    # component: list (list of indices of lines which are connected)
    for e in neighbours:
        if e not in component:
            component.append(e)


def line_from_connected_lines(line_dict, component):
    "COMPONENT contains the dictionary indices of the lines which are part of the connected component COMPONENT. This function returns the line resulting from the connected lines."
    # line_dict: Python dictionary containing the original line segments returned by the LSD algorithm
    # component: list (list of indices of lines which are connected)
    k = []
    for e in component:
        k.append(line_dict[e]["line_seg"][0:2])
        k.append(line_dict[e]["line_seg"][2:4])
    xsort = sorted(k, key=sort_by_first)
    ysort = sorted(k, key=sort_by_second)
    if LA.norm(xsort[0] - xsort[-1]) >= LA.norm(ysort[0] - ysort[-1]):
        return np.append(xsort[0], xsort[-1])
    else:
        return leftpoint(np.append(ysort[0], ysort[-1]))


def inactivate_lines(line_dict, line_indices):
    "Flags lines which are part of a connected component (i.e. of a longer line) in the line dictionary. The flagged lines won't be considered as edges as they are part of a longer line and we'd rather consider the long line"
    # line_dict: Python dictionary containing the original line segments returned by the LSD algorithm
    # line_indices: list of line indices which shall be flagged
    for e in line_indices:
        line_dict[e]["status"] = 0


################################################################################################################################################################################
# Section: Identify edge candidates
# Purpose: For each line contained in the dictionary of lines we have calculated its utility. The purpose of below functions is to create a sorted list of lines with decreasing utilities
# for each of the four different sides of the document (top, right, bottom, left).
# The first elements  in such a list would represent the top edge candidates of the respective side
################################################################################################################################################################################


def sort_by_first(item):
    # Auxiliary function of sorted() to sort for the first element in a list of lists
    return item[0]

def sort_by_second(item):
    # Auxiliary function of sorted() to sort for the second element in a list of lists
    return item[1]

def sort_by_third(item):
    # Auxiliary function of sorted() to sort for the third element in a list of lists
    return item[2]

def sort_by_forth(item):
    # Auxiliary function of sorted() to sort for the fourth element in a list of lists
    return item[3]


"""
def dict_to_array(dict):
    n = len(dict)
    if dict:
        array = np.array([dict[0]["line_seg"]])
        for i in range(1, n):
            array = np.concatenate((array, np.array([dict[i]["line_seg"]])))
        return array
"""

def line_dict_to_list(line_dict):
    "Transforms the line dictionary into a list of lists which can be sorted"
    # line_dict: Python dictionary containing the line segments
    line_list = []
    for e in line_dict:
        line_list.append([e, line_dict[e]["line_seg"], line_dict[e]["location"][1], line_dict[e]["utility"], line_dict[e]["status"]])
    return line_list

def sort_and_filter_list(line_list, location):
    "Filter the list of lines for those lines with a specific location and status (not part of connected component) and sort the filtered list according to the utility of the lines in descending order"
    # line_list: list of lists
    # location: string in {top, right, bottom, left}
    filtered_list = [item for item in line_list if item[2]==location and item[4]==1]
    return sorted(filtered_list, key = sort_by_forth, reverse=True)




################################################################################################################################################################################
# Section: Quadrilateral defined by lines
# Purpose: Together, a top, a right, a bottom and a left line, can be seen as parts of the sides of a quadrilateral. Given four line segments (top, right, bottom, left), below functions
# identify the quadrilateral spanned by these lines and calculate key characteristics of this quadrilateral
################################################################################################################################################################################

def get_quadrilateral(line_dict,edge_indices, img):
    "Given four lines, this function returns the quadrilateral spanned by the lines together with key characteristics of the quadrilateral"
    # line_dict: Python dictionary containing the line segments
    # edge_indices: 4x1 list of line indices (integers) in the line dictionary

    # Calculate the 4 vertices
    topleft = line_intersection(line_dict,edge_indices[3],edge_indices[0])
    topright = line_intersection(line_dict,edge_indices[0],edge_indices[1])
    bottomright = line_intersection(line_dict,edge_indices[1],edge_indices[2])
    bottomleft = line_intersection(line_dict,edge_indices[2],edge_indices[3])

    # Calculate area, perimeter, etc. of the quadrilateral
    area, perimeter, aspect_ratio, side_lengths = geometry(topleft,topright,bottomright,bottomleft)
    length = sum_of_edge_lengths(line_dict,edge_indices)

    utility_topleft = edge_fit_topleft(line_dict, edge_indices, side_lengths, topleft, edge_fit_radius, edge_fit_margin)
    utility_topright = edge_fit_topright(line_dict, edge_indices, side_lengths, topright, edge_fit_radius, edge_fit_margin)
    utility_bottomright = edge_fit_bottomright(line_dict, edge_indices, side_lengths, bottomright, edge_fit_radius, edge_fit_margin)
    utility_bottomleft = edge_fit_bottomleft(line_dict, edge_indices, side_lengths, bottomleft, edge_fit_radius, edge_fit_margin)

    color_top = get_color(line_dict,edge_indices[0],"top", img)
    color_right = get_color(line_dict, edge_indices[1], "right", img)
    color_bottom = get_color(line_dict, edge_indices[2], "bottom", img)
    color_left = get_color(line_dict, edge_indices[3], "left", img)

    vertices = [topleft, topright, bottomright, bottomleft]
    area = [area, utility_area(area)]
    perimeter = [perimeter, utility_perimeter(perimeter,length)]
    aspect_ratio = [aspect_ratio, utility_aspect_ration(aspect_ratio)]
    edge_fit = [utility_topleft, utility_topright, utility_bottomright, utility_bottomleft, (utility_topleft + utility_topright + utility_bottomright + utility_bottomleft) / 4]
    color = [color_top, color_right, color_bottom, color_left, utility_color(color_top, color_right, color_bottom, color_left)]
    total_utility = utility_total([area[1],perimeter[1],aspect_ratio[1],edge_fit[4],color[4]], utility_weights)
    return [vertices,area,perimeter,aspect_ratio,edge_fit,color,total_utility]

def line_intersection(line_dict, index_a, index_b):
    "Returns the intersection of two lines"
    # line_dict: Python dictionary containing the line segments
    # index_a: dictionary index of line a
    # index_b: dictionary index of line b
    line_a = line_dict[index_a]["line_seg"]
    line_b = line_dict[index_b]["line_seg"]
    term1 = (line_a[0]*line_a[3] - line_a[1]*line_a[2])
    term2 = line_b[0] - line_b[2]
    term3 = line_a[0] - line_a[2]
    term4 = (line_b[0]*line_b[3] - line_b[1]*line_b[2])
    term5 = line_b[1] - line_b[3]
    term6 = line_a[1] - line_a[3]
    return np.array([(term1 * term2 - term3 * term4)/(term3*term5 - term6*term2), (term1 * term5 - term6 * term4)/(term3*term5 - term6*term2)])

def geometry(x1, x2, x3, x4):
    "Returns area, perimeter, approx. aspect ration and side lengths of a quadrilateral with vertices x1, ..., x4"
    # xi: 2x1 numpy array representing a vertex
    top_l = LA.norm(x2-x1)
    right_l = LA.norm(x3-x2)
    bottom_l = LA.norm(x4-x3)
    left_l = LA.norm(x1-x4)
    horizontal = top_l + bottom_l
    vertical = right_l + left_l

    area = 1/2 * abs((x1[1] - x3[1]) * (x2[0] - x4[0]) + (x4[1] - x2[1]) * (x1[0] - x3[0]))
    perimeter = horizontal + vertical
    aspect_ratio = vertical / horizontal
    side_lengths = [top_l, right_l, bottom_l, left_l]

    return area, perimeter, aspect_ratio, side_lengths

def sum_of_edge_lengths(line_dict,edge_indices):
    "Returns the sum of lengths of a set of line segments"
    # line_dict: Python dictionary containing the line segments
    # edge_indices: list of indices (integers) of the line segments of which the sum of their lengths is calculated
    length = 0
    for e in edge_indices:
        length += line_dict[e]["length"][0]
    return length

def get_color(line_dict, edge_index, location, img):
    "Checks if the average color intensity of the pixels around a line segment has already been calculated and returns the color intensity"
    # line_dict: Python dictionary containing the line segments
    # edge_index: integer (index of line segment in the line dictionary)
    if "avg_color" in line_dict[edge_index]:
        return line_dict[edge_index]["avg_color"]
    else:
        vector = line_dict[edge_index]["line_seg"][2:4] - line_dict[edge_index]["line_seg"][0:2]
        line_dict[edge_index]["avg_color"] = avg_color(img,line_dict[edge_index]["line_seg"],vector,line_dict[edge_index]["length"][0],location,width)
        return line_dict[edge_index]["avg_color"]




################################################################################################################################################################################
# Section: Quadrilateral defined by lines
# Subsection: Utility of quadrilateral
# Purpose: Calculate the utility of a quadrilateral based on its area, perimeter, aspect_ratio, the fit of the line segments and the variance of the color intensities of the line segments. \
#  The utility is a measure used to identify the best quadrilateral among a set of quadrilaterals
################################################################################################################################################################################

def utility_color(color_top, color_right, color_bottom, color_left):
    ""
    if color_top and color_right and color_bottom and color_left:
        delta = max(color_top, color_right, color_bottom, color_left) - min(color_top, color_right, color_bottom, color_left)
        return 1 - delta/255
    else:
        return 0


def utility_area(area):
    max_area = region["max_paper"][0]*region["max_paper"][1]
    min_area = max_area * min_size_ratio**2
    if area > max_area:
        return 1 - (area-max_area)/max_area
    elif area < min_area:
        return 1 - (min_area-area)/min_area
    return 1


def utility_perimeter(perimeter, edge_lengths):
    return 1 - abs(perimeter - edge_lengths) / perimeter


def utility_aspect_ration(aspect_ratio):
    return 1 - abs(aspect_ratio - paper_aspect_ratio) / paper_aspect_ratio


def utility_total(utilities, weights):
    utility = 0
    for i in range(0,len(utilities)):
        utility += utilities[i] * weights[i]
    return utility


def edge_fit_topleft(line_dict, edge_indices, side_lengths, topleft_vertex, edge_fit_radius, edge_fit_margin):
    utility = 0
    top = line_dict[edge_indices[0]]["line_seg"][0:2]
    left = line_dict[edge_indices[3]]["line_seg"][0:4]
    if left[1] < left[3]:
        left = left[0:2]
    else:
        left = left[2:4]

    vec1 = top - topleft_vertex
    len_vec1 = LA.norm(vec1)
    vec2 = left - topleft_vertex
    len_vec2 = LA.norm(vec2)

    if len_vec1 < edge_fit_radius:
        utility += 1
    elif vec1[0] > 0:
        utility += 1 - (len_vec1 - edge_fit_radius) / side_lengths[0]
    elif len_vec1 - edge_fit_radius < edge_fit_margin:
        utility += 1 - 2*(len_vec1 - edge_fit_radius) / edge_fit_margin
    else:
        utility += -1

    if len_vec2 < edge_fit_radius:
        utility += 1
    elif vec2[1] > 0:
        utility += 1 - (len_vec2 - edge_fit_radius) / side_lengths[3]
    elif len_vec2 - edge_fit_radius < edge_fit_margin:
        utility += 1 - 2*(len_vec2 - edge_fit_radius) / edge_fit_margin
    else:
        utility += -1

    return utility / 2


def edge_fit_topright(line_dict, edge_indices, side_lengths, topright_vertex, edge_fit_radius, edge_fit_margin):
    utility = 0
    top = line_dict[edge_indices[0]]["line_seg"][2:4]
    right = line_dict[edge_indices[1]]["line_seg"][0:4]
    if right[1] < right[3]:
        right = right[0:2]
    else:
        right = right[2:4]

    vec1 = top - topright_vertex
    len_vec1 = LA.norm(vec1)
    vec2 = right - topright_vertex
    len_vec2 = LA.norm(vec2)

    if len_vec1 < edge_fit_radius:
        utility += 1
    elif vec1[0] < 0:
        utility += 1 - (len_vec1 - edge_fit_radius) / side_lengths[0]
    elif len_vec1 - edge_fit_radius < edge_fit_margin:
        utility += 1 - 2*(len_vec1 - edge_fit_radius) / edge_fit_margin
    else:
        utility += -1

    if len_vec2 < edge_fit_radius:
        utility += 1
    elif vec2[1] > 0:
        utility += 1 - (len_vec2 - edge_fit_radius) / side_lengths[1]
    elif len_vec2 - edge_fit_radius < edge_fit_margin:
        utility += 1 - 2*(len_vec2 - edge_fit_radius) / edge_fit_margin
    else:
        utility += -1

    return utility / 2


def edge_fit_bottomright(line_dict, edge_indices, side_lengths, bottomright_vertex, edge_fit_radius, edge_fit_margin):
    utility = 0
    bottom = line_dict[edge_indices[2]]["line_seg"][2:4]
    right = line_dict[edge_indices[1]]["line_seg"][0:4]
    if right[1] > right[3]:
        right = right[0:2]
    else:
        right = right[2:4]

    vec1 = bottom - bottomright_vertex
    len_vec1 = LA.norm(vec1)
    vec2 = right - bottomright_vertex
    len_vec2 = LA.norm(vec2)

    if len_vec1 < edge_fit_radius:
        utility += 1
    elif vec1[0] < 0:
        utility += 1 - (len_vec1 - edge_fit_radius) / side_lengths[2]
    elif len_vec1 - edge_fit_radius < edge_fit_margin:
        utility += 1 - 2*(len_vec1 - edge_fit_radius) / edge_fit_margin
    else:
        utility += -1

    if len_vec2 < edge_fit_radius:
        utility += 1
    elif vec2[1] < 0:
        utility += 1 - (len_vec2 - edge_fit_radius) / side_lengths[1]
    elif len_vec2 - edge_fit_radius < edge_fit_margin:
        utility += 1 - 2*(len_vec2 - edge_fit_radius) / edge_fit_margin
    else:
        utility += -1

    return utility / 2


def edge_fit_bottomleft(line_dict, edge_indices, side_lengths, bottomleft_vertex, edge_fit_radius, edge_fit_margin):
    utility = 0
    bottom = line_dict[edge_indices[2]]["line_seg"][0:2]
    left = line_dict[edge_indices[3]]["line_seg"][0:4]
    if left[1] > left[3]:
        left = left[0:2]
    else:
        left = left[2:4]

    vec1 = bottom - bottomleft_vertex
    len_vec1 = LA.norm(vec1)
    vec2 = left - bottomleft_vertex
    len_vec2 = LA.norm(vec2)

    if len_vec1 < edge_fit_radius:
        utility += 1
    elif vec1[0] > 0:
        utility += 1 - (len_vec1 - edge_fit_radius) / side_lengths[2]
    elif len_vec1 - edge_fit_radius < edge_fit_margin:
        utility += 1 - 2*(len_vec1 - edge_fit_radius) / edge_fit_margin
    else:
        utility += -1

    if len_vec2 < edge_fit_radius:
        utility += 1
    elif vec2[1] < 0:
        utility += 1 - (len_vec2 - edge_fit_radius) / side_lengths[3]
    elif len_vec2 - edge_fit_radius < edge_fit_margin:
        utility += 1 - 2*(len_vec2 - edge_fit_radius) / edge_fit_margin
    else:
        utility += -1

    return utility / 2




def build_quadrilater_dict(sorted_list_top, sorted_list_right, sorted_list_bottom, sorted_list_left, line_dict, img):
    quad_dict = {}
    j = 0
    for t in range(0, min(3,len(sorted_list_top))):
        for r in range(0, min(3, len(sorted_list_right))):
            for b in range(0, min(3, len(sorted_list_bottom))):
                for l in range(0, min(3, len(sorted_list_left))):
                    edge_indices = [sorted_list_top[t][0], sorted_list_right[r][0], sorted_list_bottom[b][0], sorted_list_left[l][0]]
                    quad = get_quadrilateral(line_dict, edge_indices, img)
                    quad_dict[j] = {"vertices": quad[0],
                                    "area": quad[1],
                                    "perimeter": quad[2],
                                    "aspect_ratio": quad[3],
                                    "edge_fit": quad[4],
                                    "color": quad[5],
                                    "total_utility": quad[6],
                                    "edge_indices": edge_indices}
                    j+=1
    return quad_dict





def quad_dict_to_list(quad_dict):
    quad_list = []
    for e in quad_dict:
        quad_list.append([e, quad_dict[e]["vertices"], quad_dict[e]["total_utility"], quad_dict[e]["edge_indices"]])
    return quad_list

def sort_quad_list(quad_list):
    return sorted(quad_list, key = sort_by_third, reverse=True)


def original_vertices(vertices, resize_factor):
    return [resize_factor*item for item in vertices]














def document_vertices(img_original):

    size_original = img_original.shape[:2]

    img = resize_image(img_original, target_size)
    img_size = img.shape[:2]
    resize_factor = size_original[0] / img_size[0]
    region = main_region(img_size, region_parameters)

    lines = line_segment_detector(img)
    line_dict = build_line_dictionary(lines, img_size)

    adj_matrix = adjacency_matrix(line_dict, radius, rho)
    connected_components(adj_matrix, line_dict, img_size)

    line_list = line_dict_to_list(line_dict)
    sorted_list_top = sort_and_filter_list(line_list, "top")
    sorted_list_left = sort_and_filter_list(line_list, "left")
    sorted_list_right = sort_and_filter_list(line_list, "right")
    sorted_list_bottom = sort_and_filter_list(line_list, "bottom")

    quad_dict = build_quadrilater_dict(sorted_list_top, sorted_list_right, sorted_list_bottom, sorted_list_left, line_dict, img)
    quad_list = quad_dict_to_list(quad_dict)
    quad_list = sort_quad_list(quad_list)

    if quad_list[0][2] > 0.7:
        vertices = original_vertices(quad_list[0][1],resize_factor)
        print("The vertices of the document in the image are: ",vertices)
        return vertices
    else:
        vertices = [np.array([region["x_bounds"][0] * resize_factor, region["y_bounds"][0] * resize_factor]),
                    np.array([region["x_bounds"][3] * resize_factor, region["y_bounds"][0] * resize_factor]),
                    np.array([region["x_bounds"][3] * resize_factor, region["y_bounds"][3] * resize_factor]),
                    np.array([region["x_bounds"][0] * resize_factor, region["y_bounds"][3] * resize_factor])]
        print("Please manually adjust the vertices as the document edges could not be identified.")
        return vertices


def warp_image(vertices, img_original):
    height = int(max(abs(vertices[3][1] - vertices[0][1]), abs(vertices[2][1] - vertices[1][1])))
    width = int(height / paper_aspect_ratio)
    pts1 = np.float32([vertices[0], vertices[1], vertices[2], vertices[3]])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(img_original, M, (width, height))
    return img_warped







################################################################################################################################################################################
################################################################################################################################################################################
#                                                                   Run!
################################################################################################################################################################################
################################################################################################################################################################################


# Parameters Definitions

_refine = 1
_scale =0.6
_sigma_scale = 0.6
_quant = 1.5
_ang_th = 22.5
_log_eps = 0.0
_density_th = 0.7
_n_bins = 1024

paper_aspect_ratio = np.sqrt(2)
min_size_ratio = 3/5
margin = 0.6

region_parameters = [paper_aspect_ratio, min_size_ratio, margin]

target_size = 380
radius = 5
rho = 5

# Color scan
width = 2

# Edge Fit
edge_fit_radius = 8
edge_fit_margin = 8

# Utility Quadrilateral
utility_weights = [0.2, 0.2, 0.2, 0.2, 0.2]



# Load an color image in grayscale
img_original = cv2.imread('scan (23).jpg',0)
img = resize_image(img_original,target_size)
img_size = img.shape[:2]
region = main_region(img_size,region_parameters)


vertices = document_vertices(img_original)

img_warped = warp_image(vertices,img_original)

img_warped = cv2.medianBlur(img_warped,5)
img_warped = cv2.adaptiveThreshold(img_warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,7)
print(img_warped)

cv2.imwrite('warped.png',img_warped)
cv2.imshow('image',img_warped)
cv2.waitKey(0)
cv2.destroyAllWindows()




"""
img_original[int(vertices[0][1]):int(vertices[0][1]) + 25, int(vertices[0][0]):int(vertices[0][0]) + 25] = 255
img_original[int(vertices[1][1]):int(vertices[1][1]) + 25, int(vertices[1][0]) - 25:int(vertices[1][0])] = 255
img_original[int(vertices[2][1]) - 25:int(vertices[2][1]), int(vertices[2][0]) - 25:int(vertices[2][0])] = 255
img_original[int(vertices[3][1]) - 25:int(vertices[3][1]), int(vertices[3][0]):int(vertices[3][0]) + 25] = 255
cv2.imshow('image',img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()




size_original = img_original.shape[:2]

#img = cv2.GaussianBlur(img,(49,49),0)

img = resize_image(img_original,target_size)
img_size = img.shape[:2]

region = main_region(img_size,region_parameters)


'''
img = cv2.pyrDown(img_original)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
'''

# height, width = img.shape[:2]


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


start = timeit.default_timer()
lines = line_segment_detector(img)
print("Initial number of lines: ",len(lines))



line_dict  = build_line_dictionary(lines, img_size)
print("Reduced number of lines: ", len(line_dict))
end = timeit.default_timer()
print(end - start)




start = timeit.default_timer()
adjMatrix = adjacency_matrix(line_dict, radius, rho)
end = timeit.default_timer()
print(end - start)
#print(adjMatrix)


connected_components(adjMatrix, line_dict, img_size)


list = line_dict_to_list(line_dict)


print("Sorted Top: ", sort_and_filter_list(list,"top"))
print("Sorted Right: ", sort_and_filter_list(list,"right"))
print("Sorted Bottom: ", sort_and_filter_list(list,"bottom"))
print("Sorted Left: ", sort_and_filter_list(list,"left"))
sorted_list_top = sort_and_filter_list(list,"top")
sorted_list_left = sort_and_filter_list(list,"left")
sorted_list_right = sort_and_filter_list(list,"right")
sorted_list_bottom = sort_and_filter_list(list,"bottom")

quad_dict = build_quadrilater_dict(sorted_list_top, sorted_list_right, sorted_list_bottom, sorted_list_left, line_dict, img)
quad_list = quad_dict_to_list(quad_dict)
quad_list = sort_quad_list(quad_list)
print(quad_list)

def lines_from_edge_indices(edge_indices):
    return np.array([[line_dict[edge_indices[0]]["line_seg"]], [line_dict[edge_indices[1]]["line_seg"]], [line_dict[edge_indices[2]]["line_seg"]], [line_dict[edge_indices[3]]["line_seg"]]])

for e in quad_list:
    img2 = np.copy(img)
    img2[int(e[1][0][1]):int(e[1][0][1]) + 5, int(e[1][0][0]):int(e[1][0][0]) + 5] = 255
    img2[int(e[1][1][1]):int(e[1][1][1]) + 5, int(e[1][1][0]) - 5:int(e[1][1][0])] = 255
    img2[int(e[1][2][1]) - 5:int(e[1][2][1]), int(e[1][2][0]) - 5:int(e[1][2][0])] = 255
    img2[int(e[1][3][1]) - 5:int(e[1][3][1]), int(e[1][3][0]):int(e[1][3][0]) + 5] = 255
    edges = lines_from_edge_indices(e[3])
    img2 = cv2.createLineSegmentDetector().drawSegments(img2, edges)
    print("Total Utility: ", e[2])
    print("Utility Area: ", quad_dict[e[0]]["area"])
    print("Utility Perimeter: ", quad_dict[e[0]]["perimeter"])
    print("Utility Aspect Ratio: ", quad_dict[e[0]]["aspect_ratio"])
    print("Utility Edge Fit: ", quad_dict[e[0]]["edge_fit"])
    print("Utility Color: ", quad_dict[e[0]]["color"])

    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    del img2


"""