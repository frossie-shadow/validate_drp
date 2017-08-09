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

import operator

import astropy.units as u
from matplotlib import pyplot as plt
import numpy as np
import treecorr

from lsst.validate.base import MeasurementBase, Metric

from ..util import (averageRaFromCat, averageDecFromCat,
                    medianEllipticity1ResidualsFromCat,
                    medianEllipticity2ResidualsFromCat)


class TExMeasurement(MeasurementBase):
    """Measurement of TEx (x=1,2): Correlation of PSF residual ellipticity
    on scales of D=(1, 5) arcmin.

    Parameters
    ----------
    metric : `lsst.validate.base.Metric`
        An TE1 or TE2 `~lsst.validate.base.Metric` instance.
    matchedDataset : lsst.validate.drp.matchreduce.MatchedMultiVisitDataset
        The matched catalogs to analyze.
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
    The TEx table below is provided in ``validate_drp``\ 's :file:`metrics.yaml`.

    LPM-17 dated 2011-07-06

    Specification:
        Using the full survey data, the E1, E2, and EX residual PSF ellipticity
        correlations averaged over an arbitrary FOV must have the median
        less than TE1 for theta <= 1 arcmin, and less than TE2 for theta >= 5 arcmin.

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
        self.register_parameter('bin_range_operator', datum=self.metric.bin_range_operator)

        # Place to save correlation function vs radius
        # for later plotting or re-analysis
        self.register_extra('radius', label='Correlation radius', unit=u.arcmin)
        self.register_extra('xip', label='Correlation strength', unit=u.Unit(''))
        self.register_extra('xip_err', label='Correlation strength uncertainty', unit=u.Unit(''))

        # Add external blob so that links will be persisted with
        # the measurement
        if linkedBlobs is not None:
            for name, blob in linkedBlobs.items():
                setattr(self, name, blob)

        matches = matchedDataset.safeMatches

        self.radius, self.xip, self.xip_err = \
            correlation_function_ellipticity_from_matches(matches, verbose=verbose)

        corr, corr_err = select_bin_from_corr(
            self.radius,
            self.xip,
            self.xip_err,
            radius=self.D,
            operator=Metric.convert_operator_str(self.bin_range_operator))

        # We store the absolute value.
        self.quantity = np.abs(corr) * u.Unit('')

        if job:
            job.register_measurement(self)


def correlation_function_ellipticity_from_matches(matches, **kwargs):
    """Compute shear-shear correlation function for ellipticity residual from a 'MatchedMultiVisitDataset' object.

    Convenience function for calling correlation_function_ellipticity.

    Parameters
    ----------
    matches : `lsst.validate.drp.matchreduce.MatchedMultiVisitDataset`
        - The matched catalogs to analyze.

    Returns
    -------
    r, xip, xip_err : each a np.array(dtype=float)
        - The bin centers, two-point correlation, and uncertainty.
    """
    ra = matches.aggregate(averageRaFromCat) * u.radian
    dec = matches.aggregate(averageDecFromCat) * u.radian

    e1_res = matches.aggregate(medianEllipticity1ResidualsFromCat)
    e2_res = matches.aggregate(medianEllipticity2ResidualsFromCat)

    return correlation_function_ellipticity(ra, dec, e1_res, e2_res, **kwargs)


def correlation_function_ellipticity(ra, dec, e1_res, e2_res,
                                     nbins=20, min_sep=0.25, max_sep=20,
                                     sep_units='arcmin', verbose=False):
    """Compute shear-shear correlation function from ra, dec, g1, g2.

    Default parameters for nbins, min_sep, max_sep chosen to cover
       an appropriate range to calculate TE1 (<=1 arcmin) and TE2 (>=5 arcmin).
    Parameters
    ----------
    ra : numpy.array
        Right ascension of points [radians]
    dec : numpy.array
        Declination of points [radians]
    e1_res : numpy.array
        Residual ellipticity 1st component
    e2_res : numpy.array
        Residual ellipticity 2nd component
    nbins : float, optional
        Number of bins over which to analyze the two-point correlation
    min_sep : float, optional
        Minimum separation over which to analyze the two-point correlation
    max_sep : float, optional
        Maximum separation over which to analyze the two-point correlation
    sep_units : str, optional
        Specify the units of min_sep and max_sep
    verbose : bool
        Request verbose output from `treecorr`.
        verbose=True will use verbose=2 for `treecorr.GGCorrelation`.

    Returns
    -------
    r, xip, xip_err : each a np.array(dtype=float)
        - The bin centers, two-point correlation, and uncertainty.
    """
    # Translate to 'verbose_level' here to refer to the integer levels in TreeCorr
    # While 'verbose' is more generically what is being passed around
    #   for verbosity within 'validate_drp'
    if verbose:
        verbose_level = 2
    else:
        verbose_level = 0

    catTree = treecorr.Catalog(ra=ra, dec=dec, g1=e1_res, g2=e2_res,
                               dec_units='radian', ra_units='radian')
    gg = treecorr.GGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                sep_units=sep_units,
                                verbose=verbose_level)
    gg.process(catTree)
    r = np.exp(gg.meanlogr) * u.arcmin
    xip = gg.xip * u.Unit('')
    xip_err = np.sqrt(gg.varxi) * u.Unit('')

    return (r, xip, xip_err)


def select_bin_from_corr(r, xip, xip_err, radius=1, operator=operator.le):
    """Aggregate measurements for r less than (or greater than) radius.

    Returns aggregate measurement for all entries where operator(r, radius).
    E.g.,
     * Passing radius=5, operator=operator.le will return averages for r<=5
     * Passing radius=2, operator=operator.gt will return averages for r >2

    Written with the use of correlation functions in mind, thus the naming
    but generically just returns averages of the arrays xip and xip_err
    where the condition is satsified

    Parameters
    ----------
    r : numpy.array
        radius
    xip : numpy.array
        correlation
    xip_err : numpy.array
        correlation uncertainty
    operator : Operation in the 'operator' module: le, ge, lt, gt

    Returns
    -------
    avg_xip, avg_xip_err : (float, float)
    """

    w, = np.where(operator(r, radius))

    avg_xip = np.average(xip[w])
    avg_xip_err = np.average(xip_err[w])

    return avg_xip, avg_xip_err


def plot_correlation_function_ellipticity(r, xip, xip_err,
                                          plotfile='ellipticity_corr.png'):
    """
    Parameters
    ----------
    r : numpy.array
        Correlation average radius in each bin
    xip : numpy.array
        Two-point correlation
    xip_err : numpy.array
        Uncertainty on two-point correlation

    Effects
    -------
    Creates a plot file in the local filesystem: 'ellipticty_corr.png'
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(r.value, xip, yerr=xip_err)
    ax.set_xlabel('Separation (arcmin)', size=19)
    ax.set_ylabel('Median Residual Ellipticity Correlation', size=19)
    fig.savefig(plotfile)
