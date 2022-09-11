#!/usr/bin/env python
import numpy as np
from scipy.linalg import expm, logm
from lab5_header import *

"""
Angles are in radian, distance are in meters.
"""
def Get_MS():
	# Fill in the correct values for S1~6, as well as the M matrix
	M = np.array([[0,-1,0,390e-3],[0,0,-1,401e-3],[1,0,0,215.5e-3],[0,0,0,1]])
	q_list = [np.array([-150e-3,150e-3,10e-3]),np.array([-150e-3,270e-3,162e-3]),np.array([94e-3,270e-3,162e-3]),np.array([307e-3,187e-3,162e-3]),np.array([307e-3,270e-3,162e-3]),np.array([390e-3,270e-3,162e-3])]
	w_list = [np.array([0,0,1]),np.array([0,1,0]),np.array([0,1,0]),np.array([0,1,0]),np.array([1,0,0]),np.array([0,1,0])]
	v_list = []
	S_list = []
	for n in range(len(q_list)):
		v_list.append(-np.cross(w_list[n],q_list[n]))

	for n in range(len(q_list)):
		S_list.append(np.array([[0,-w_list[n][2],w_list[n][1],v_list[n][0]],[w_list[n][2],0,-w_list[n][0],v_list[n][1]],[-w_list[n][1],w_list[n][0],0,v_list[n][2]],[0,0,0,0]]))

	return M, S_list


"""
Function that calculates encoder numbers for each motor
"""
def lab_fk(theta1, theta2, theta3, theta4, theta5, theta6):

	# Initialize the return_value
	return_value = [None, None, None, None, None, None]

	theta_list = [theta1,theta2,theta3,theta4,theta5,theta6]
	M, S_list = Get_MS()
	exp_list = []
	for n in range(len(S_list)):
		exp_list.append(expm(S_list[n]*theta_list[n]))

	Tbe = np.matmul(exp_list[0],np.matmul(exp_list[1],np.matmul(exp_list[2],np.matmul(exp_list[3],np.matmul(exp_list[4],np.matmul(exp_list[5],M))))))

	print(str(Tbe) + "\n")

	return_value[0] = theta1 + PI
	return_value[1] = theta2
	return_value[2] = theta3
	return_value[3] = theta4 - (0.5*PI)
	return_value[4] = theta5
	return_value[5] = theta6

	print(np.degrees(return_value))
	return return_value

def lab_invk(xWgrip, yWgrip, zWgrip, yaw_WgripDegree):
	yaw_WgripRads = (yaw_WgripDegree/180)*PI
	L = [-1,0.152,0.120,0.244,0.093,0.213,0.083,0.083,0.082,0.0535,0.059]
	xWgrip += 0.15
	yWgrip -= 0.15
	zWgrip -= 0.01

	xcen = xWgrip - (L[9]*np.cos(yaw_WgripRads))
	ycen = yWgrip - (L[9]*np.sin(yaw_WgripRads))
	zcen = zWgrip

	theta1 = np.arctan2(ycen,xcen) - np.arcsin((L[6] + 0.027)/np.sqrt(xcen**2 + ycen**2))

	x3end = xcen + (L[6] + 0.027)*np.sin(theta1) - L[7]*np.cos(theta1) 
	y3end = ycen - (L[6] + 0.027)*np.cos(theta1) - L[7]*np.sin(theta1) 
	z3end = zcen + L[8] + L[10]
	
	a = np.sqrt(((z3end - L[1])**2) + (x3end**2) + (y3end**2))

	theta2 = (-1*np.arccos(((L[3])**2 + (a**2) - (L[5])**2)/(2*a*(L[3])))) + (-1*np.arcsin((z3end - L[1])/a))
	theta3 = PI - np.arccos(((L[3]**2) + (L[5]**2) - (a**2))/(2*L[3]*L[5]))
	theta4 = ((-1*theta2) + (-1*theta3))
	theta5 = -PI/2
	theta6 = (theta1 + PI/2 - yaw_WgripRads)
	
	return lab_fk(theta1,theta2,theta3,theta4,theta5,theta6)
