#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from scipy.optimize import fmin_slsqp, fmin_bfgs
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitscost, distanceToObstacle, jointlimitsviolated
from pinocchio.utils import rotate

import time
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON, DT
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET


# def eq_constraint(q):
#     closest_valid_q = projecttojointlimits(robot, q)
#     q_distance_to_joint_range = np.linalg.norm(q - closest_valid_q)
#     print(q_distance_to_joint_range)
#     return np.array([q_distance_to_joint_range])
    # return distanceToObstacle(robot, q)

def ineq_constraint(q):
    distance = np.abs(distanceToObstacle(robot, q))# - 100*collision(robot,q)
    return np.array([distance])
    # return np.concatenate((q_upper_limit, q_lower_limit, np.array([distance])), axis=None)

def bfgs_minimisation_objective(q, cube_placement):
    oMleft_cube, oMright_cube = cube_placement
    
    pin.framesForwardKinematics(robot.model, robot.data, q)

    oMleft_effector=robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
    left_effectorMleft_cube = oMleft_effector.inverse() * oMleft_cube

    oMright_effector=robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
    right_effectorMright_cube = oMright_effector.inverse() * oMright_cube

    left_cost = pin.log(right_effectorMright_cube).vector
    right_cost = pin.log(left_effectorMleft_cube).vector

    return (
            10*(np.linalg.norm(left_cost, ord=2)
            + np.linalg.norm(right_cost, ord=2))
            + 100*ineq_constraint(q)
            + 1000*(jointlimitscost(robot,q))
    )

def slsqp_minimisation_objective(q, cube_placement):
    oMleft_cube, oMright_cube = cube_placement
    
    pin.framesForwardKinematics(robot.model, robot.data, q)

    oMleft_effector=robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
    left_effectorMleft_cube = oMleft_effector.inverse() * oMleft_cube

    oMright_effector=robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
    right_effectorMright_cube = oMright_effector.inverse() * oMright_cube

    left_cost = pin.log(right_effectorMright_cube).vector
    right_cost = pin.log(left_effectorMleft_cube).vector

    # Choosing the 1-norm over the 2-norm seems to improve convergence and state space exploration.
    # - I believe this makes sense given that we are looking for a sparse solution
    # as well as the fact that the L2 norm promotes similarity of the decision variables.
    # The set of optimal solutions is going to be tiny (and sparse!) Compared to the feasible region.
    return (
            (np.linalg.norm(left_cost, ord=1)
            + np.linalg.norm(right_cost, ord=1))
    )

def optimiser_callback(q):
    updatevisuals(viz, robot, cube, q)

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    oMleft_cube=getcubeplacement(cube, LEFT_HOOK)
    oMright_cube=getcubeplacement(cube, RIGHT_HOOK)

    joint_bounds = [(robot.model.lowerPositionLimit[i], robot.model.upperPositionLimit[i]) for i in range(0, len(robot.model.upperPositionLimit))]

    # bfgs_t0 = time.time()
    # bfgs_result = fmin_bfgs(lambda q: bfgs_minimisation_objective(q, (oMleft_cube, oMright_cube)), qcurrent, callback=optimiser_callback, full_output=True, disp=False)
    # bfgs_t1 = time.time()
    # bfgs_xopt,bfgs_fopt,_,bfgs_gopt,_,bfgs_warnflag,_ = bfgs_result
    
    slsqp_t0 = time.time()
    slsqp_result = fmin_slsqp(lambda q: slsqp_minimisation_objective(q, (oMleft_cube, oMright_cube)), qcurrent,  f_ieqcons=ineq_constraint, bounds=joint_bounds,  callback=optimiser_callback, full_output=True, disp=False,iter=1000)
    slsqp_t1 = time.time()
    slsqp_out,slsqp_fx,_,slsqp_imode,slsqp_smode = slsqp_result

    # print("bfgs: ", bfgs_t1 - bfgs_t0, bfgs_fopt, bfgs_warnflag)
    print("slsqp: ", slsqp_t1 - slsqp_t0, slsqp_fx, slsqp_imode)

    return slsqp_out, True if slsqp_imode == 0 and slsqp_fx < EPSILON else False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()

    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    # Randomised cube positions for testing arbitrary cube positions
    for i in range(5):
        qi, success = computeqgrasppose(robot, q, cube, pin.SE3(rotate('z', np.random.rand()), np.array( [np.random.rand()-0.5, np.random.rand()-0.5,  0.94]) ) )
    
    updatevisuals(viz, robot, cube, qe)
    
    
    
