#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

Two tasks:
- See what optimal values can be found based on initial configuration or initial configuration with robot looking behind (just rotated 180 degrees)
- Test to see if an optimal solution can be found faster and more accurately by generating 50 initial configurations and lowering the tolerance.

@author: stonneau & Finbar
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from scipy.optimize import fmin_slsqp, fmin_bfgs, minimize
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
    distance = distanceToObstacle(robot, q)# - 100*collision(robot,q)
    return np.array([20*distance])
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

    # Given that the changes in the y axis poses do not affect the minima (i.e changing them results in different configurations that are in the null space)
    # 
    # left_cost[4] = 0
    # right_cost[4] = 0

    lower = robot.model.lowerPositionLimit
    upper = robot.model.upperPositionLimit   

    viol_low  = np.maximum(0.0, lower - q)
    viol_high = np.maximum(0.0, q - upper)
    limit_penalty = 1*np.sum(viol_low**2 + viol_high**2)

    distance = distanceToObstacle(robot, q)
    penetration = np.maximum(0.0, - distance)
    collision_penalty = 50 * penetration


    return (
            (np.linalg.norm(left_cost, ord=2)
            + np.linalg.norm(right_cost, ord=2))
            # + collision_penalty + limit_penalty
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

    # These allow us ignore the y pose of the end effector
    # left_cost[4] = 0
    # right_cost[4] = 0

    # Choosing the 1-norm over the 2-norm seems to improve convergence and state space exploration.
    # - I believe this makes sense given that we are looking for a sparse solution
    # as well as the fact that the L2 norm promotes similarity of the decision variables.
    # The set of optimal solutions is going to be tiny (and sparse!) Compared to the feasible region.
    return (
            # When using 2 norm solutions are found more often but slower, while 1 norm can find sparser solutions faster
            (np.linalg.norm(left_cost, ord=2)
            + np.linalg.norm(right_cost, ord=2))
            # + 0.01*np.linalg.norm(q[0], ord=2)
            + 5*np.linalg.norm(q[1:3], ord=2)
            # + 0.0001*np.linalg.norm(q, ord=2)
    )

def optimiser_callback(q):
    pass
    updatevisuals(viz, robot, cube, q)
    time.sleep(0.1)

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    oMleft_cube=getcubeplacement(cube, LEFT_HOOK)
    oMright_cube=getcubeplacement(cube, RIGHT_HOOK)

    q_reversed = robot.q0
    q_reversed[0] += np.pi


    # bfgs_t0 = time.time()
    # bfgs_result = fmin_bfgs(lambda q: bfgs_minimisation_objective(q, (oMleft_cube, oMright_cube)), qcurrent, gtol=1e-10, callback=optimiser_callback, full_output=True, disp=False)
    # bfgs_t1 = time.time()
    # bfgs_xopt,bfgs_fopt,_,bfgs_gopt,_,bfgs_warnflag,_ = bfgs_result
    # print("bfgs: ", bfgs_t1 - bfgs_t0, bfgs_fopt, bfgs_warnflag)
    # print("Distance of robot to configuration: ", slsqp_minimisation_objective(bfgs_xopt, (oMleft_cube, oMright_cube)))
    #

    joint_bounds = [(robot.model.lowerPositionLimit[i], robot.model.upperPositionLimit[i]) for i in range(0, len(robot.model.upperPositionLimit))]

    # slsqp_t0 = time.time()
    # slsqp_result = fmin_slsqp(lambda q: slsqp_minimisation_objective(q, (oMleft_cube, oMright_cube)), qcurrent,  f_ieqcons=ineq_constraint, bounds=joint_bounds,  callback=optimiser_callback, full_output=True, disp=False,iter=3000)
    # slsqp_t1 = time.time()
    # slsqp_out,slsqp_fx,_,slsqp_imode,slsqp_smode = slsqp_result
    # print("slsqp: ", slsqp_t1 - slsqp_t0, slsqp_fx, slsqp_imode)
    # print("Distance of robot to configuration: ", slsqp_minimisation_objective(slsqp_out, (oMleft_cube, oMright_cube)))

    slsqp_t0 = time.time()
    slsqp_result = minimize(
                            lambda q: slsqp_minimisation_objective(q, 
                            (oMleft_cube, oMright_cube)), 
                            qcurrent,  
                            method='SLSQP', 
                            constraints={
                                'type': 'ineq', 
                                'fun': ineq_constraint
                            }, 
                            bounds=joint_bounds,  
                            callback=optimiser_callback, 
                            tol=1e-10,
                            options= {'maxiter': 200}
    )
    print(slsqp_result)

    # slsqp_t1 = time.time()
    # slsqp_out,slsqp_fx,_,slsqp_imode,slsqp_smode = slsqp_result
    # print("slsqp: ", slsqp_t1 - slsqp_t0, slsqp_fx, slsqp_imode)
    # print("Distance of robot to configuration: ", slsqp_minimisation_objective(slsqp_out, (oMleft_cube, oMright_cube)))


    # print("SWAPPING HANDS")
    # bfgs_t0 = time.time()
    # bfgs_result = fmin_bfgs(lambda q: bfgs_minimisation_objective(q, (oMright_cube, oMleft_cube)), qcurrent, gtol=1e-10, callback=optimiser_callback, full_output=True, disp=False)
    # bfgs_t1 = time.time()
    # bfgs_xopt,bfgs_fopt,_,bfgs_gopt,_,bfgs_warnflag,_ = bfgs_result
    # print("bfgs: ", bfgs_t1 - bfgs_t0, bfgs_fopt, bfgs_warnflag)
    # print("Distance of robot to configuration: ", slsqp_minimisation_objective(bfgs_xopt, (oMleft_cube, oMright_cube)))
    #
    # joint_bounds = [(robot.model.lowerPositionLimit[i], robot.model.upperPositionLimit[i]) for i in range(0, len(robot.model.upperPositionLimit))]
    #
    # slsqp_t0 = time.time()
    # slsqp_result = fmin_slsqp(lambda q: slsqp_minimisation_objective(q, (oMright_cube, oMleft_cube)), bfgs_xopt,  f_ieqcons=ineq_constraint, bounds=joint_bounds,  callback=optimiser_callback, full_output=True, disp=False,iter=3000)
    # slsqp_t1 = time.time()
    # slsqp_out,slsqp_fx,_,slsqp_imode,slsqp_smode = slsqp_result
    # print("slsqp: ", slsqp_t1 - slsqp_t0, slsqp_fx, slsqp_imode)
    # print("Distance of robot to configuration: ", slsqp_minimisation_objective(slsqp_out, (oMright_cube, oMleft_cube)))

    # For reversed configuration (doesn't really do anything tbh)
    # slsqp_t0 = time.time()
    # slsqp_result = fmin_slsqp(lambda q: slsqp_minimisation_objective(q, (oMleft_cube, oMright_cube)), q_reversed,  f_ieqcons=ineq_constraint, bounds=joint_bounds,  callback=optimiser_callback, full_output=True, disp=False,iter=3000)
    # slsqp_t1 = time.time()
    # slsqp_out,slsqp_fx,_,slsqp_imode,slsqp_smode = slsqp_result
    # print("slsqp: ", slsqp_t1 - slsqp_t0, slsqp_fx, slsqp_imode)
    # print("Distance of robot to configuration: ", slsqp_minimisation_objective(slsqp_out, (oMleft_cube, oMright_cube)))


    print("iteration: ", slsqp_minimisation_objective(slsqp_result.x, (oMleft_cube, oMright_cube)), slsqp_result.success)

    return slsqp_result.x, True if slsqp_result.success and slsqp_minimisation_objective(slsqp_result.x, (oMleft_cube, oMright_cube)) < EPSILON else False
    # return slsqp_out, True if slsqp_imode == 0 and slsqp_minimisation_objective(slsqp_out, (oMleft_cube, oMright_cube)) < EPSILON else False
    # return bfgs_xopt, True if bfgs_warnflag == 0 and slsqp_minimisation_objective(bfgs_xopt, (oMleft_cube, oMright_cube)) < EPSILON else False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()

    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    print("Success? ", successinit)
    time.sleep(2)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    print("Success? ", successend)

    # Randomised cube positions for testing arbitrary cube positions
    for i in range(5):
        qi, success = computeqgrasppose(robot, q, cube, pin.SE3(rotate('z', np.random.rand()), np.array( [np.random.rand()-0.5, np.random.rand()-0.5,  0.94]) ) )
        print("Success? ", success)
        time.sleep(1)
    
    updatevisuals(viz, robot, cube, qe)
    
    
    
