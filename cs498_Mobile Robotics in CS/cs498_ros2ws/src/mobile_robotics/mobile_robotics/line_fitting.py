import numpy as np
import matplotlib.pyplot as plt
import math


def polar_to_xy(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def estimate_coef(x, y):
    '''
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    '''
    b_1 = (y[0] - y[1]) / (x[0] - x[1])
    b_0 = y[0] - b_1 * x[0]
  
    return (b_0, b_1)

def shortest_distance(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d


# test data in polar coordinates
rho_test = np.array([[10, 11, 11.7, 13, 14, 15, 16, 17, 17, 17, 16.5, 17, 17, 16, 14.5, 14, 13]]).T
n = rho_test.shape[0]
theta_test = (math.pi/180) * np.linspace(0, 85, n).reshape(-1,1)


# convert polar coordinates to xy coordinates
x, y = polar_to_xy(rho_test, theta_test)
x = np.reshape(x, (x.shape[0]))
y = np.reshape(y, (y.shape[0]))
plt.scatter(x, y)


'''
Split and Merge algorithm
'''
# initialize the first set to contain all the points
first_set_x = x
first_set_y = y 
# initialize the queue
points_x_queue = []
points_y_queue = []
points_x_queue.append(first_set_x)
points_y_queue.append(first_set_y)
# initialize the list to store parameters of selected lines
# every element is a tuple: (x_coordinates of vertices, y_coordinates of vertices, line parameters)
lines_para = []

dist_thres = 0.5
while len(points_x_queue) != 0 and len(points_y_queue) != 0:
    cur_set_x = points_x_queue.pop(0)
    cur_set_y = points_y_queue.pop(0)

    # fit the line for two extreme points6
    x1 = cur_set_x[0]
    y1 = cur_set_y[0]
    x2 = cur_set_x[cur_set_x.shape[0] - 1]
    y2 = cur_set_y[cur_set_y.shape[0] - 1]
    x_ext = np.array([x1, x2])
    y_ext = np.array([y1, y2])

    b = estimate_coef(x_ext, y_ext)  # b[0] is b, b[1] is the slope

    # find the greatest distance to the line
    max_dist = -1
    split_idx = -1
    for i in range(cur_set_x.shape[0]):
        dist = shortest_distance(cur_set_x[i], cur_set_y[i], b[1], -1, b[0])
        if dist > max_dist:
            max_dist = dist
            split_idx = i
    
    # determine whether to split the set
    if max_dist > dist_thres:
        # split the set
        subset_1_x = cur_set_x[0:split_idx + 1]
        subset_1_y = cur_set_y[0:split_idx + 1]
        subset_2_x = cur_set_x[split_idx:]
        subset_2_y = cur_set_y[split_idx:]
        points_x_queue.append(subset_1_x)
        points_y_queue.append(subset_1_y)
        points_x_queue.append(subset_2_x)
        points_y_queue.append(subset_2_y)
    else:
        # the line is good, store the line parameters
        lines_para.append((x_ext, y_ext, b))
        plt.plot(x_ext, y_ext, color='r')


plt.show()
        





