import copy
import unittest

import numpy as np
from numpy import testing

import skrobot
from skrobot.model.primitives import Box
from easyopt import TinyfkSweptSphereSdfCollisionChecker
from easyopt.utils import set_robot_config


def jacobian_test_util(func, x0, decimal=5):
    f0, jac = func(x0)
    n_dim = len(x0)

    eps = 1e-7
    jac_numerical = np.zeros(jac.shape)
    for idx in range(n_dim):
        x1 = copy.copy(x0)
        x1[idx] += eps
        f1, _ = func(x1)
        jac_numerical[:, idx] = (f1 - f0) / eps

    print(jac_numerical.T)
    print(jac.T)
    testing.assert_almost_equal(jac, jac_numerical, decimal=decimal)


class TestCollisionChecker(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        robot_model = skrobot.models.PR2()
        robot_model.init_pose()

        table = Box(extents=[0.7, 1.0, 0.05], with_sdf=True)
        table.translate([0.8, 0.0, 0.655])

        link_idx_table = {}
        for link_idx in range(len(robot_model.link_list)):
            name = robot_model.link_list[link_idx].name
            link_idx_table[name] = link_idx

        coll_link_names = ["r_upper_arm_link"]

        coll_link_list = [robot_model.link_list[link_idx_table[lname]]
                          for lname in coll_link_names]

        link_names = ["r_shoulder_pan_link", "r_shoulder_lift_link",
                      "r_upper_arm_roll_link", "r_elbow_flex_link",
                      "r_forearm_roll_link", "r_wrist_flex_link",
                      "r_wrist_roll_link"]

        link_list = [robot_model.link_list[link_idx_table[lname]]
                     for lname in link_names]
        joint_list = [l.joint for l in link_list]

        coll_checker = TinyfkSweptSphereSdfCollisionChecker(table.sdf, robot_model)
        for link in coll_link_list:
            coll_checker.add_collision_link(link)
        joint_ids = coll_checker.fksolver.get_joint_ids([j.name for j in joint_list])

        cls.robot_model = robot_model
        cls.coll_checker = coll_checker
        cls.joint_list = joint_list
        cls.joint_ids = joint_ids

    def test_coll_batch_forward_kinematics(self):
        robot_model = self.robot_model
        av = np.array([0.4, 0.6] + [-0.7] * 5)
        av_with_base = np.hstack((av, [0.1, 0.0, 0.3]))
        joint_list = self.joint_list
        joint_ids = self.joint_ids
        set_robot_config(robot_model, joint_list, av_with_base, with_base=True)

        n_wp = 1
        n_feature = self.coll_checker.n_feature
        print("n_feature :{0}".format(n_feature))
        print("coll :{0}".format(self.coll_checker.coll_coords_list))

        def collision_fk(av, with_base):
            # this function wraps _coll_batch_forward_kinematics so that it's
            # output will be scipy-style (f, jac)
            n_dof = len(av)
            with_rot = False
            with_jacobian = True
            P_tmp, J_tmp = self.coll_checker.fksolver.solve_forward_kinematics(
                np.array([av]), self.coll_checker.coll_sphere_id_list, joint_ids,
                with_rot, with_base, with_jacobian)
            f = P_tmp.flatten()
            jac = J_tmp.reshape(n_wp, n_feature, 3, n_dof).reshape(n_wp*n_feature*3, n_dof)
            return f, jac

        jacobian_test_util(lambda av: collision_fk(av, False), av)
        jacobian_test_util(lambda av: collision_fk(av, True), av_with_base)

    def test_compute_batch_sd_vals(self):
        robot_model = self.robot_model
        joint_list = self.joint_list
        av = np.array([0.4, 0.6] + [-0.7] * 5)
        set_robot_config(robot_model, joint_list, av)
        joint_names = [j.name for j in joint_list]
        joint_ids = self.coll_checker.fksolver.get_joint_ids(joint_names)

        def func(av):
            return self.coll_checker._compute_batch_sd_vals(
                joint_ids, np.array([av]), with_jacobian=True)
        jacobian_test_util(func, av)
