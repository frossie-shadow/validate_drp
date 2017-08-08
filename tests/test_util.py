# LSST Data Management System
# Copyright 2017 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

from __future__ import division, print_function

import unittest

import lsst.utils

from lsst.validate.drp import util


class UtilCalculations(lsst.utils.tests.TestCase):
    """Test utility functions."""

    def testEllipticityHorizonalLine(self):
        """Is util.ellipticity correct for a horizontal line."""

        Ixx, Ixy, Iyy = 1, 0, 0
        exp_e, exp_e1, exp_e2 = 1+0j, 1, 0

        obs_e, obs_e1, obs_e2 = util.ellipticity(Ixx, Ixy, Iyy)
        self.assertFloatsAlmostEqual(exp_e, obs_e)
        self.assertFloatsAlmostEqual(exp_e1, obs_e1)
        self.assertFloatsAlmostEqual(exp_e2, obs_e2)

    def testEllipticityVerticalLine(self):
        """Is util.ellipticity correct for a vertical line."""

        Ixx, Ixy, Iyy = 0, 0, 1
        exp_e, exp_e1, exp_e2 = -1+0j, -1, 0

        obs_e, obs_e1, obs_e2 = util.ellipticity(Ixx, Ixy, Iyy)
        self.assertFloatsAlmostEqual(exp_e, obs_e)
        self.assertFloatsAlmostEqual(exp_e1, obs_e1)
        self.assertFloatsAlmostEqual(exp_e2, obs_e2)

    def testEllipticityDiagonalLine(self):
        """Is util.ellipticity correct for a diagonal line."""

        Ixx, Ixy, Iyy = 1, 1, 1
        exp_e, exp_e1, exp_e2 = 0.5+0.5j, 0.5, 0.5

        obs_e, obs_e1, obs_e2 = util.ellipticity(Ixx, Ixy, Iyy)
        print(obs_e, obs_e1, obs_e2)
        self.assertFloatsAlmostEqual(exp_e, obs_e)
        self.assertFloatsAlmostEqual(exp_e1, obs_e1)
        self.assertFloatsAlmostEqual(exp_e2, obs_e2)

    def testEllipticityCircle(self):
        """Is util.ellipticity correct for a circle."""

        Ixx, Ixy, Iyy = 1, 0, 1
        exp_e, exp_e1, exp_e2 = 0+0j, 0, 0

        obs_e, obs_e1, obs_e2 = util.ellipticity(Ixx, Ixy, Iyy)
        self.assertFloatsAlmostEqual(exp_e, obs_e)
        self.assertFloatsAlmostEqual(exp_e1, obs_e1)
        self.assertFloatsAlmostEqual(exp_e2, obs_e2)

    def testEllipticityEllipse(self):
        """Is util.ellipticity correct for an ellipse."""

        Ixx, Ixy, Iyy = 4, 0, 1
        exp_e, exp_e1, exp_e2 = 1/3+0j, 1/3, 0

        obs_e, obs_e1, obs_e2 = util.ellipticity(Ixx, Ixy, Iyy)
        self.assertFloatsAlmostEqual(exp_e, obs_e)
        self.assertFloatsAlmostEqual(exp_e1, obs_e1)
        self.assertFloatsAlmostEqual(exp_e2, obs_e2)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
