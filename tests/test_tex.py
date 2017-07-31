# LSST Data Management System
# Copyright 2012-2017 LSST Corporation.
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

import operator
import unittest

import numpy as np
import numpy.random as random

import astropy.units as u

import lsst.utils
from lsst.validate.drp.calcsrd.tex import select_bin_from_corr, correlation_function_ellipticity


class TexCalculations(lsst.utils.tests.TestCase):
    """Test calculation of TEx ellipticity residuals calculations."""

    def testSelectBinFromCorr(self):
        """Does select_bin_from_corr correctly return only and all the bins that satisfy condition."""
        r = np.array([0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5])
        xip = 1e-5 * np.array([0.9, 1.1, 1.5, 0.02, 1.2, 10.0, 5.0, 2.5, 3.0])
        xip_err = 1e-6 * np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])

        exp_all_avg_xip, exp_all_avg_xip_err = np.average(xip), np.average(xip_err)
        obs_all_avg_xip, obs_all_avg_xip_err = \
            select_bin_from_corr(r, xip, xip_err, radius=0, operator=operator.gt)
        self.assertFloatsAlmostEqual(exp_all_avg_xip, obs_all_avg_xip, rtol=1e-7)
        self.assertFloatsAlmostEqual(exp_all_avg_xip_err, obs_all_avg_xip_err, rtol=1e-7)

        exp_rlt2_avg_xip, exp_rlt2_avg_xip_err = 2.8171429e-05, 1.2857143e-06
        obs_rlt2_avg_xip, obs_rlt2_avg_xip_err = \
            select_bin_from_corr(r, xip, xip_err, radius=2, operator=operator.lt)
        self.assertFloatsAlmostEqual(exp_rlt2_avg_xip, obs_rlt2_avg_xip, rtol=1e-7)
        self.assertFloatsAlmostEqual(exp_rlt2_avg_xip_err, obs_rlt2_avg_xip_err, rtol=1e-7)

        exp_rge1_avg_xip, exp_rge1_avg_xip_err = 4.3400000e-05, 1.8000000e-06
        obs_rge1_avg_xip, obs_rge1_avg_xip_err = \
            select_bin_from_corr(r, xip, xip_err, radius=1, operator=operator.ge)
        self.assertFloatsAlmostEqual(exp_rge1_avg_xip, obs_rge1_avg_xip, rtol=1e-7)
        self.assertFloatsAlmostEqual(exp_rge1_avg_xip_err, obs_rge1_avg_xip_err, rtol=1e-7)

    def testEllipticityResidualCorr(self):
        """Does the correlation function correctly compute for a random field?

        Our goal is mostly to check that we're calling this function correctly.
        Tests of the deeper performance are implicitly outsourced to TreeCorr.

        Use same approach as TreeCorr galaxy-galaxy test from
        https://github.com/rmjarvis/TreeCorr/blob/releases/3.3/tests/test_gg.py

        Which in turn references
        http://adsabs.harvard.edu/abs/2002A%26A...389..729S
        """
        rand_seed = 1238625876
        random.seed(rand_seed)
        # Yes, a million.  N this high and L this large
        #  gets xip within 1e-7 absolute of the analytic limit.
        # Takes 25 seconds to run on an early-2015 MacBook Air 2.2 GHz Intel Core i7
        N = 1000000
        L = 500 * u.arcmin
        ra = ((random.random_sample(N)-0.5) * L).to(u.rad)
        dec = ((random.random_sample(N)-0.5) * L).to(u.rad)

        r0 = 10 * u.arcmin
        gamma0 = 0.05

        # Ignoring spherical geometry cos(dec) term
        x, y = ra.to(u.arcmin), dec.to(u.arcmin)
        r2 = (x**2 + y**2)/r0**2

        g1 = -gamma0 * np.exp(-r2/2) * (x**2-y**2)/r0**2
        g2 = -gamma0 * np.exp(-r2/2) * (2*x*y)/r0**2

        obs_r, obs_xip, obs_xip_err = \
            correlation_function_ellipticity(ra, dec, g1, g2)

        r = obs_r
        prefactor = np.pi/16 * gamma0**2 * (r0/L)**2 * np.exp(-0.25*(r/r0)**2)
        exp_xip = prefactor * (r**4 - 16*r**2 * r0**2 + 32*r0**4)/r0**4

        self.assertFloatsAlmostEqual(exp_xip, obs_xip, atol=1e-7, rtol=1e-1)

        # 2017-08-05 MWV:
        #  I don't know how to calculate the expected xip_err
        #  so there's presently no test for that.


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
