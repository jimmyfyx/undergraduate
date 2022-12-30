# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP3
"""

import math
import numpy as np
from alien import Alien
from typing import List, Tuple

def does_alien_touch_wall(alien, walls,granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    offset = granularity / math.sqrt(2)
    for wall in walls:
        line = ((wall[0], wall[1]), (wall[2], wall[3]))
        if alien.is_circle() == True:
            # circle
            if point_segment_distance(alien.get_centroid(), line) < alien.get_width() + offset or np.isclose(point_segment_distance(alien.get_centroid(), line), alien.get_width() + offset):
                return True
        else:
            # sausage
            alien_line = (alien.get_head_and_tail()[0], alien.get_head_and_tail()[1])
            if segment_distance(alien_line, line) < alien.get_width() + offset or np.isclose(segment_distance(alien_line, line), alien.get_width() + offset):
                return True
        
    return False


def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be mImplementing Convolutional Neural Networksultiple goals
        
        Return:
            True if a goal is touched, False if not.
    """
    for goal in goals:
        if alien.is_circle() == True:
            # circle
            if math.sqrt((alien.get_centroid()[0] - goal[0])**2 + (alien.get_centroid()[1] - goal[1])**2) < alien.get_width() + goal[2] or np.isclose(math.sqrt((alien.get_centroid()[0] - goal[0])**2 + (alien.get_centroid()[1] - goal[1])**2), alien.get_width() + goal[2]):
                return True
        else:
            # sausage
            alien_line = (alien.get_head_and_tail()[0], alien.get_head_and_tail()[1])
            if point_segment_distance((goal[0], goal[1]), alien_line) < alien.get_width() + goal[2] or np.isclose(point_segment_distance((goal[0], goal[1]), alien_line), alien.get_width() + goal[2]):
                return True

    return False


def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    line_1 = (0, 0, window[0], 0)
    line_2 = (0, window[1], window[0], window[1])
    line_3 = (0, 0, 0, window[1])
    line_4 = (window[0], 0, window[0], window[1])
    boundary_list = [line_1, line_2, line_3, line_4]
    return not does_alien_touch_wall(alien, boundary_list, granularity)


def point_segment_distance(point, segment):
    """Compute the distance from the point to the line segment.
    Hint: Lecture note "geometry cheat sheet"

        Args:
            point: A tuple (x, y) of the coordinates of the point.
            segment: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    vec_ae = (point[0] - segment[0][0], point[1] -segment[0][1])
    vec_ab = (segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])
    vec_be = (point[0] - segment[1][0], point[1] - segment[1][1])

    dot_ab_be = vec_ab[0] * vec_be[0] + vec_ab[1] * vec_be[1]
    dot_ab_ae = vec_ab[0] * vec_ae[0] + vec_ab[1] * vec_ae[1]

    if dot_ab_be > 0:
        return math.sqrt(vec_be[0] ** 2 + vec_be[1] ** 2)
    elif dot_ab_ae < 0:
        return math.sqrt(vec_ae[0] ** 2 + vec_ae[1] ** 2)
    else:
        ab_norm = math.sqrt(vec_ab[0]**2 + vec_ab[1]**2)
        if ab_norm == 0:
            return math.sqrt(vec_ae[0] ** 2 + vec_ae[1] ** 2)
        else:
            dist = (vec_ae[0] * vec_ab[1] - vec_ae[1] * vec_ab[0]) / ab_norm
            return abs(dist)

"""
The idea and part of the codes (do_segments_intersects) give credit to Algorithm Tutor:
https://algorithmtutor.com/Computational-Geometry/Check-if-two-line-segment-intersect/
"""
def direction(p1, p2, p3):
    first = (p3[0] - p1[0], p3[1] - p1[1])
    second = (p2[0] - p1[0], p2[1] - p1[1])
    return first[0] * second[1] - second[0] * first[1]

def on_segment(p1, p2, p):
    return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])

def do_segments_intersect(segment1, segment2):
    """Determine whether segment1 intersects segment2.  
    We recommend implementing the above first, and drawing down and considering some examples.
    Lecture note "geometry cheat sheet" may also be handy.

        Args:
            segment1: A tuple of coordinates indicating the endpoints of segment1.
            segment2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    p1 = segment1[0]
    p2 = segment1[1]
    p3 = segment2[0]
    p4 = segment2[1]

    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    elif d1 == 0 and on_segment(p3, p4, p1):
        return True
    elif d2 == 0 and on_segment(p3, p4, p2):
        return True
    elif d3 == 0 and on_segment(p1, p2, p3):
        return True
    elif d4 == 0 and on_segment(p1, p2, p4):
        return True
    else:
        return False


def segment_distance(segment1, segment2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.
    Hint: Distance of two line segments is the distance between the closest pair of points on both.

        Args:
            segment1: A tuple of coordinates indicating the endpoints of segment1.
            segment2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(segment1, segment2) == True:
        return 0
    
    min_dist = float('inf')
    for points2 in segment2:
        for points1 in segment1:
            cur_dist = point_segment_distance(points2, segment1)
            if cur_dist < min_dist:
                min_dist = cur_dist
    
    for points1 in segment1:
        for points2 in segment2:
            cur_dist = point_segment_distance(points1, segment2)
            if cur_dist < min_dist:
                min_dist = cur_dist

    return min_dist
    

if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result

    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                  f'{b} is expected to be {result[i]}, but your' \
                                                                  f'result is {distance}'

    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()
        
        touch_wall_result = does_alien_touch_wall(alien, walls, 0)
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert touch_goal_result == truths[
            1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, ' \
                f'expected: {truths[1]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    # Initialize Aliens and perform simple sanity check.
    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
