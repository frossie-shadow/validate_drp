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

data_path = os.path.join(os.getenv('VALIDATION_DATA_CFHT_DIR'), 'data')
print(data_path)
butler = Butler(data_path)

r_visits= [849375, 850587]


xip=[]
xip_err=[]

def calculate_ellipticity(shape):
    I_xx, I_xy, I_yy = shape.getIxx(), shape.getIxy(), shape.getIyy()
    e = (I_xx - I_yy + 2j*I_xy) / (I_xx+I_yy + 2*np.sqrt(I_xx*I_yy - I_xy*2))
    e1 = np.imag(e)
    e2 = np.real(e)
    return e, e1, e2

for visit in r_visits:
    print visit
    e1_res = []
    e2_res = []
    ra = []
    dec = []
    for i in range(104):

        try:
            src = butler.get('src',visit=visit, ccd=i)
            img = butler.get('calexp',visit=visit, ccd=i,flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            psf = img.getPsf()
        except:
            continue
        print i

        for s in src:
            psf_shape = psf.computeShape(s.getCentroid())
            psf_e, psf_e1, psf_e2 = calculate_ellipticity(psf_shape)
            star_shape = s.getShape()
            star_e, star_e1, star_e2 = calculate_ellipticity(star_shape)
            if np.isfinite([star_e1, psf_e1, star_e2, psf_e2]).all():
                ra.append(s.getCoord().getRa().asDegrees())
                dec.append(s.getCoord().getDec().asDegrees())
                e1_res.append(star_e1-psf_e1)
                e2_res.append(star_e2-psf_e2)

    print np.mean(e1_res),np.mean(e2_res)
    if np.isnan(np.mean(e1_res)):
        print 'error'
        continue
    catTree = treecorr.Catalog(ra=ra, dec=dec, g1=e1_res, g2=e2_res,
                               dec_units='degree', ra_units='degree')
    gg = treecorr.GGCorrelation(nbins=10, min_sep=0.5, max_sep=10, sep_units='arcmin',
                                verbose=2)
    gg.process(catTree)
    r = np.exp(gg.meanlogr)
    xip.append(gg.xip)
    xip_err.append(np.sqrt(gg.varxi))

xip=np.array(xip)
med_xip = [np.median(xip[:,i]) for i in range(len(xip[0]))]
err_xip = [np.std(xip[:,i])/np.sqrt(len(xip)) for i in range(len(xip[0]))]
print r
print med_xip
print err_xip

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(r, med_xip, yerr=err_xip)
ax.set_xlabel('Separation (arcmin)',size=19)
ax.set_ylabel('Median Residual Ellipticity Correlation',size=19)
fig.show()
