# LSST Data Management System
# Copyright 2016 AURA/LSST.
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

from __future__ import print_function, absolute_import
from builtins import zip

import numpy as np
import astropy.units as u

from lsst.validate.base import MeasurementBase

from ..util import (averageRaDecFromCat, averageRaFromCat, averageDecFromCat,
                    sphDist)


class AMxMeasurement(MeasurementBase):
    """Measurement of AMx (x=1,2,3): The maximum rms of the astrometric
    distance distribution for stellar pairs with separations of D arcmin
    (repeatability).

    Parameters
    ----------
    metric : `lsst.validate.base.Metric`
        An AM1, AM2 or AM3 `~lsst.validate.base.Metric` instance.
    matchedDataset : lsst.validate.drp.matchreduce.MatchedMultiVisitDataset
    filter_name : `str`
        filter_name (filter name) used in this measurement (e.g., ``'r'``).
    width : `float` or `astropy.units.Quantity`, optional
        Width around fiducial distance to include. [arcmin]
    magRange : 2-element `list`, `tuple`, or `numpy.ndarray`, optional
        brighter, fainter limits of the magnitude range to include.
        Default: ``[17.5, 21.5]`` mag.
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
    rmsDistMas : ndarray
        RMS of distance repeatability between stellar pairs.
    blob : AMxBlob
        Blob with by-products from this measurement.

    Notes
    -----
    This table below is provided ``validate_drp``\ 's :file:`metrics.yaml`.

    LPM-17 dated 2011-07-06

    Specification:
        The rms of the astrometric distance distribution for
        stellar pairs with separation of D arcmin (repeatability)
        will not exceed AMx milliarcsec (median distribution for a large number
        of sources). No more than AFx % of the sample will deviate by more than
        ADx milliarcsec from the median. AMx, AFx, and ADx are specified for
        D=5, 20 and 200 arcmin for x= 1, 2, and 3, in the same order (Table 18).

    The three selected characteristic distances reflect the size of an
    individual sensor, a raft, and the camera. The required median astrometric
    precision is driven by the desire to achieve a proper motion accuracy of
    0.2 mas/yr and parallax accuracy of 1.0 mas over the course of the survey.
    These two requirements correspond to relative astrometric precision for a
    single image of 10 mas (per coordinate).

    ========================= ====== ======= =======
    Astrometric Repeatability          Specification
    ------------------------- ----------------------
                       Metric Design Minimum Stretch
    ========================= ====== ======= =======
            AM1 (milliarcsec)     10      20       5
            AF1 (%)               10      20       5
            AD1 (milliarcsec)     20      40      10
            AM2 (milliarcsec)     10      20       5
            AF2 (%)               10      20       5
            AD2 (milliarcsec)     20      40      10
            AM3 (milliarcsec)     15      30      10
            AF3 (%)               10      20       5
            AD3 (milliarcsec)     30      50      20
    ========================= ====== ======= =======

    Table 18: The specifications for astrometric precision.
    The three blocks of values correspond to D=5, 20 and 200 arcmin,
    and to astrometric measurements performed in the r and i bands.
    """

    def __init__(self, metric, matchedDataset, filter_name, width=2.,
                 magRange=None, linkedBlobs=None, job=None, verbose=False):
        MeasurementBase.__init__(self)

        self.metric = metric
        self.filter_name = filter_name

        # Register blob
        self.matchedDataset = matchedDataset

        # Measurement Parameters
        self.register_parameter('D', datum=self.metric.D)

        if not isinstance(width, u.Quantity):
            width = width * u.arcmin
        self.register_parameter('width',
                                quantity=width,
                                label='Width',
                                description='Width of annulus')
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

        annulus = self.D + (self.width/2)*np.array([-1, +1])
        self.register_parameter('annulus',
                                quantity=annulus,
                                label='annulus radii',
                                description='Inner and outer radii of '
                                            'selection annulus.')

        # Register measurement extras
        self.register_extra('rmsDistMas', label='RMS')

        # Add external blob so that links will be persisted with
        # the measurement
        if linkedBlobs is not None:
            for name, blob in linkedBlobs.items():
                setattr(self, name, blob)

        matches = matchedDataset.safeMatches
        rmsDistances = calcRmsDistances(
            matches,
            self.annulus,
            magRange=self.magRange,
            verbose=verbose)

        if len(rmsDistances) == 0:
            # raise ValidateErrorNoStars(
            #     'No stars found that are %.1f--%.1f arcmin apart.' %
            #     (annulus[0], annulus[1]))
            # FIXME should we still report that this measurement was
            # attempted instead of just crashing.
            print('No stars found that are {0:.1f}--{1:.1f} apart.'.format(
                  self.annulus[0], self.annulus[1]))
            self.rmsDistMas = None
            self.quantity = None
        else:
            self.rmsDistMas = rmsDistances.to(u.marcsec)
            self.quantity = np.median(self.rmsDistMas)

        if job:
            job.register_measurement(self)


def calcRmsDistances(groupView, annulus, magRange, verbose=False):
    """Calculate the RMS distance of a set of matched objects over visits.

    Parameters
    ----------
    groupView : lsst.afw.table.GroupView
        GroupView object of matched observations from MultiMatch.
    annulus : length-2 `astropy.units.Quantity`
        Distance range (i.e., arcmin) in which to compare objects.
        E.g., `annulus=np.array([19, 21]) * u.arcmin` would consider all
        objects separated from each other between 19 and 21 arcminutes.
    magRange : length-2 `astropy.units.Quantity`
        Magnitude range from which to select objects.
    verbose : bool, optional
        Output additional information on the analysis steps.

    Returns
    -------
    rmsDistances : `astropy.units.Quantity`
        RMS angular separations of a set of matched objects over visits.
    """

    # First we make a list of the keys that we want the fields for
    importantKeys = [groupView.schema.find(name).key for
                     name in ['id', 'coord_ra', 'coord_dec',
                              'object', 'visit', 'base_PsfFlux_mag']]

    minMag, maxMag = magRange.to(u.mag).value

    def magInRange(cat):
        mag = cat.get('base_PsfFlux_mag')
        w, = np.where(np.isfinite(mag))
        medianMag = np.median(mag[w])
        return minMag <= medianMag and medianMag < maxMag

    groupViewInMagRange = groupView.where(magInRange)

    # List of lists of id, importantValue
    matchKeyOutput = [obj.get(key)
                      for key in importantKeys
                      for obj in groupViewInMagRange.groups]

    jump = len(groupViewInMagRange)

    ra = matchKeyOutput[1*jump:2*jump]
    dec = matchKeyOutput[2*jump:3*jump]
    visit = matchKeyOutput[4*jump:5*jump]

    # Calculate the mean position of each object from its constituent visits
    # `aggregate` calulates a quantity for each object in the groupView.
    meanRa = groupViewInMagRange.aggregate(averageRaFromCat)
    meanDec = groupViewInMagRange.aggregate(averageDecFromCat)

    annulusRadians = arcminToRadians(annulus.to(u.arcmin).value)

    rmsDistances = list()
    for obj1, (ra1, dec1, visit1) in enumerate(zip(meanRa, meanDec, visit)):
        dist = sphDist(ra1, dec1, meanRa[obj1+1:], meanDec[obj1+1:])
        objectsInAnnulus, = np.where((annulusRadians[0] <= dist) &
                                     (dist < annulusRadians[1]))
        for obj2 in objectsInAnnulus:
            distances = matchVisitComputeDistance(
                visit[obj1], ra[obj1], dec[obj1],
                visit[obj2], ra[obj2], dec[obj2])
            if not distances:
                if verbose:
                    print("No matching visits found for objs: %d and %d" %
                          (obj1, obj2))
                continue

            finiteEntries, = np.where(np.isfinite(distances))
            if len(finiteEntries) > 0:
                rmsDist = np.std(np.array(distances)[finiteEntries])
                rmsDistances.append(rmsDist)

    # return quantity
    rmsDistances = np.array(rmsDistances) * u.radian
    return rmsDistances


def matchVisitComputeDistance(visit_obj1, ra_obj1, dec_obj1,
                              visit_obj2, ra_obj2, dec_obj2):
    """Calculate obj1-obj2 distance for each visit in which both objects are seen.

    For each visit shared between visit_obj1 and visit_obj2,
    calculate the spherical distance between the obj1 and obj2.

    visit_obj1 and visit_obj2 are assumed to be unsorted.

    Parameters
    ----------
    visit_obj1 : scalar, list, or numpy.array of int or str
        List of visits for object 1.
    ra_obj1 : scalar, list, or numpy.array of float
        List of RA in each visit for object 1.  [radians]
    dec_obj1 : scalar, list or numpy.array of float
        List of Dec in each visit for object 1. [radians]
    visit_obj2 : list or numpy.array of int or str
        List of visits for object 2.
    ra_obj2 : list or numpy.array of float
        List of RA in each visit for object 2.  [radians]
    dec_obj2 : list or numpy.array of float
        List of Dec in each visit for object 2.  [radians]

    Results
    -------
    list of float
        spherical distances (in radians) for matching visits.
    """
    distances = []
    visit_obj1_idx = np.argsort(visit_obj1)
    visit_obj2_idx = np.argsort(visit_obj2)
    j_raw = 0
    j = visit_obj2_idx[j_raw]
    for i in visit_obj1_idx:
        while (visit_obj2[j] < visit_obj1[i]) and (j_raw < len(visit_obj2_idx)-1):
            j_raw += 1
            j = visit_obj2_idx[j_raw]
        if visit_obj2[j] == visit_obj1[i]:
            if np.isfinite([ra_obj1[i], dec_obj1[i],
                            ra_obj2[j], dec_obj2[j]]).all():
                distances.append(sphDist(ra_obj1[i], dec_obj1[i],
                                         ra_obj2[j], dec_obj2[j]))
    return distances


def radiansToMilliarcsec(rad):
    return np.rad2deg(rad)*3600*1000


def arcminToRadians(arcmin):
    return np.deg2rad(arcmin/60)
