# LSST Data Management System
# Copyright 2015-2017 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.

import treecorr
from lsst.daf.persistence import Butler
import lsst.afw.table as afwTable
import numpy as np
from matplotlib import pyplot as plt
import os

import astropy.units as u

from lsst.validate.base import MeasurementBase

from ..util import medianEllipticityResidualsFromCat

class TExMeasurement(MeasurementBase):
    """Measurement of TEx (x=1,2): Correlation of PSF residual ellipticity
    on scales of D=(1, 5) arcmin.

    Parameters
    ----------
    metric : `lsst.validate.base.Metric`
        An TE1 or TE2 `~lsst.validate.base.Metric` instance.
    matchedDataset : lsst.validate.drp.matchreduce.MatchedMultiVisitDataset
    filter_name : `str`
        filter_name (filter name) used in this measurement (e.g., ``'r'``).
    verbose : `bool`, optional
        Output additional information on the analysis steps.
    job : :class:`lsst.validate.drp.base.Job`, optional
        If provided, the measurement will register itself with the Job
        object.
    linkedBlobs : dict, optional
        A `dict` of additional blobs (subclasses of BlobBase) that
        can provide additional context to the measurement, though aren't
        direct dependencies of the computation (e.g., ``matchedDataset``).

    Attributes
    ----------
    blob : TExBlob
        Blob with by-products from this measurement.

    Notes
    -----
    This table below is provided ``validate_drp``\ 's :file:`metrics.yaml`.

    LPM-17 dated 2011-07-06

    Specification:
        Using the full survey data, the E1, E2, and EX residual PSF ellipticity
        correlations averaged over an arbitrary FOV must have the median
        less than TE1 for θ ≤ 1 arcmin, and less than TE2 for θ ≥ 5 arcmin.

    The residual ellipticity correlations vary smoothly so it is sufficient to
    specify limits in these two angular ranges. On 1 arcmin to 5 arcmin scales,
    these residual ellipticity correlations put LSST systematics a factor of a
    few below the weak lensing shot noise, i.e., statistical errors will
    dominate over systematics. On larger scales, the noise level imposed by
    nature due to shot noise plus cosmic variance is almost scale-independent,
    whereas the atmospheric contribution to systematics becomes negligible.
    Therefore the specifications on 5 arcmin scales apply to all larger scales
    as well (as per section 2.1.1). On scales larger than the field of view,
    sources of systematic error have less to do with the instrumentation than
    with the operations (due to the seeing distribution), software, and algorithms.

    ========================= ====== ======= =======
    PSF Ellipticity Residuals     Specification
    ------------------------- ----------------------
                       Metric Design Minimum Stretch
    ========================= ====== ======= =======
            TE1 ()             2e-5    3e-5   1e-5
            TE2 (%)            1e-7    3e-7   5e-8
            TEF (%)             15      15     10
            TE3 ()             4e-5    6e-5   2e-5
            TE4 ()             2e-7    5e-7   1e-7
    ========================= ====== ======= =======


    Table 27: These residual PSF ellipticity correlations apply to the r and i bands.
    """

    def __init__(self, metric, matchedDataset, filter_name,
                 linkedBlobs=None, job=None, verbose=False):
        MeasurementBase.__init__(self)

        self.metric = metric
        self.filter_name = filter_name

        # Register blob
        self.matchedDataset = matchedDataset

        # Measurement Parameters
        self.register_parameter('D', datum=self.metric.D)

        if magRange is None:
            magRange = np.array([17.0, 21.5]) * u.mag
        else:
            assert len(magRange) == 2
            if not isinstance(magRange, u.Quantity):
                magRange = np.array(magRange) * u.mag
        self.register_parameter('magRange',
                                quantity=magRange,
                                description='Stellar magnitude selection '
                                            'range.')

        # Register measurement extras
        self.register_extra('ellipticityCorrelation', label='ellipticity correlation')

        # Add external blob so that links will be persisted with
        # the measurement
        if linkedBlobs is not None:
            for name, blob in linkedBlobs.items():
                setattr(self, name, blob)

        matches = matchedDataset.safeMatches

        corr_in_bins = correlation_function_ellipticity(matches, circle_radius=1)
        corr = select_bin_from_corr(corr_in_bins, circle_radius=1)

        self.ellipticityCorrelation = corr_in_bins
        self.quantity = corr * u.Unit('')

        if job:
            job.register_measurement(self)

xip=[]
xip_err=[]

ra = groupViewInMagRange.aggregate(averageRaFromCat) * u.radian
dec = groupViewInMagRange.aggregate(averageDecFromCat) * u.radian
e1_res, e2_res = groupViewInMagRange.aggregate(medianEllipticityResidualsFromCat)

catTree = treecorr.Catalog(ra=ra, dec=dec, g1=e1_res, g2=e2_res,
                           dec_units='radian', ra_units='radian')
gg = treecorr.GGCorrelation(nbins=10, min_sep=0.5, max_sep=10, sep_units='arcmin',
                            verbose=2)
gg.process(catTree)
r = np.exp(gg.meanlogr)
xip = gg.xip
xip_err = np.sqrt(gg.varxi)

print(r)
print(xip)
print(xip_err)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(r, xip, yerr=xip_err)
ax.set_xlabel('Separation (arcmin)',size=19)
ax.set_ylabel('Median Residual Ellipticity Correlation',size=19)
fig.show()
