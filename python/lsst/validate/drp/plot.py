# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
"""Matplotlib plots describing lsst.validate.drp metric measurements, as well
as analytic models of photometric and astrometric repeatability.
"""

from __future__ import print_function, division

import matplotlib.pylab as plt
import numpy as np
import astropy.units as u
import scipy.stats
from .astromerrmodel import astromErrModel
from .photerrmodel import photErrModel


__all__ = ['plotOutlinedAxline',
           'plotAstrometryErrorModel',
           'plotAstromErrModelFit', 'plotPhotErrModelFit',
           'plotPhotometryErrorModel', 'plotPA1', 'plotAMx']


# Plotting defaults
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
# plt.rcParams['figure.titlesize'] = 30

color = {'all': 'grey', 'bright': 'blue',
         'iqr': 'green', 'rms': 'red'}


def plotOutlinedAxline(axMethod, x, **kwargs):
    """Plot an axis line with a white shadow for better contrast.

    Parameters
    ----------
    axMethod : `matplotlib.pyplot.axhline` or `matplotlib.pyplot.axvline`
        A horizontal or vertical axis line plotting function.
    x : float
        Axis coordinate
    **kwargs :
        Keyword arguments for `~matplotlib.pyplot.axhline` or
        `~matplotlib.pyplot.axvline`.
    """
    shadowArgs = dict(kwargs)
    foregroundArgs = dict(kwargs)

    if 'linewidth' not in foregroundArgs:
        foregroundArgs['linewidth'] = 3

    if 'linewidth' in shadowArgs:
        shadowArgs['linewidth'] += 1
    else:
        shadowArgs['linewidth'] = 4
    shadowArgs['color'] = 'w'
    shadowArgs['label'] = None

    axMethod(x, **shadowArgs)
    axMethod(x, **foregroundArgs)


def plotAstrometryErrorModel(dataset, astromModel, outputPrefix=''):
    """Plot angular distance between matched sources from different exposures.

    Creates a file containing the plot with a filename beginning with
    `outputPrefix`.

    Parameters
    ----------
    dataset : `MatchedMultiVisitDataset`
        Blob with the multi-visit photometry model.
    photomModel : `AnalyticPhotometryModel`
        An analyticPhotometry model object.
    outputPrefix : str, optional
        Prefix to use for filename of plot file.  Will also be used in plot
        titles. E.g., ``outputPrefix='Cfht_output_r_'`` will result in a file
        named ``'Cfht_output_r_check_astrometry.png'``.
    """
    bright, = np.where(dataset.snr > astromModel.brightSnr)

    numMatched = len(dataset.dist)
    dist_median = np.median(dataset.dist)
    bright_dist_median = np.median(dataset.dist[bright])

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(18, 12))

    ax[0].hist(dataset.dist, bins=100, color=color['all'],
               histtype='stepfilled', orientation='horizontal')
    ax[0].hist(dataset.dist[bright], bins=100, color=color['bright'],
               histtype='stepfilled', orientation='horizontal')

    ax[0].set_ylim([0., 500.])
    ax[0].set_ylabel("Distance [{unit:latex}]".format(unit=dataset.dist.unit))
    plotOutlinedAxline(
        ax[0].axhline,
        dist_median.value,
        color=color['all'],
        label="Median RMS: {v.value:.1f} {v.unit:latex}".format(v=dist_median))
    plotOutlinedAxline(
        ax[0].axhline,
        bright_dist_median.value,
        color=color['bright'],
        label="SNR > {snr:.0f}\nMedian RMS: {v.value:.1f} {v.unit:latex}".format(
            snr=astromModel.brightSnr,
            v=bright_dist_median))
    ax[0].legend(loc='upper right')

    ax[1].scatter(dataset.snr, dataset.dist,
                  s=10, color=color['all'], label='All')
    ax[1].scatter(dataset.snr[bright], dataset.dist[bright], s=10,
                  color=color['bright'],
                  label='SNR > {0:.0f}'.format(astromModel.brightSnr))
    ax[1].set_xlabel("SNR")
    ax[1].set_xscale("log")
    ax[1].set_ylim([0., 500.])
    matchCountTemplate = '\n'.join([
        'Matches:',
        '{nBright:d} high SNR,',
        '{nAll:d} total'])
    ax[1].text(0.6, 0.6, matchCountTemplate.format(nBright=len(bright),
                                                   nAll=numMatched),
               transform=ax[1].transAxes, ha='left', va='baseline')

    w, = np.where(dataset.dist < 200 * u.marcsec)
    plotAstromErrModelFit(dataset.snr[w], dataset.dist[w], astromModel,
                          ax=ax[1])

    ax[1].legend(loc='upper right')
    ax[1].axvline(astromModel.brightSnr,
                  color='red', linewidth=4, linestyle='dashed')
    plotOutlinedAxline(
        ax[0].axhline,
        dist_median.value,
        color=color['all'])
    plotOutlinedAxline(
        ax[0].axhline,
        bright_dist_median.value,
        color=color['bright'])

    # Using title rather than suptitle because I can't get the top padding
    plt.suptitle("Astrometry Check : %s" % outputPrefix,
                 fontsize=30)
    if outputPrefix == '':
        plotPath = "check_astrometry.png"
    else:
        plotPath = "%s_%s" % (outputPrefix, "check_astrometry.png")
    plt.savefig(plotPath, format="png")
    plt.close(fig)


def plotAstromErrModelFit(snr, dist, model,
                          color='red', ax=None, verbose=True):
    """Plot model of photometric error from LSST Overview paper
    http://arxiv.org/abs/0805.2366v4

    Astrometric Errors
    error = C * theta / SNR

    Parameters
    ----------
    snr : list or numpy.array
        S/N of photometric measurements
    dist : list or numpy.array
        Separation from reference [mas]
    model : `AnalyticAstrometryModel`
        An `AnalyticAstrometryModel` instance.
    """
    if ax is None:
        ax = plt.figure()
        xlim = [10, 30]
    else:
        xlim = ax.get_xlim()

    x_model = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), num=100)
    fit_model_mas_err = astromErrModel(x_model,
                                       theta=model.theta,
                                       sigmaSys=model.sigmaSys,
                                       C=model.C)
    ax.plot(x_model, fit_model_mas_err,
            color=color, linewidth=2,
            label='Model')

    modelLabelTemplate = '\n'.join([
        r'$C = {C:.2g}$',
        r'$\theta$ = {theta:.4g}',
        r'$\sigma_\mathrm{{sys}}$ = {sigmaSys.value:.2g} {sigmaSys.unit:latex}'])
    modelLabel = modelLabelTemplate.format(
        C=model.C,
        theta=model.theta,
        sigmaSys=model.sigmaSys)
    ax.text(0.6, 0.4, modelLabel,
            transform=ax.transAxes, va='baseline', ha='left', color=color)
    # Set the x limits back to their original values.
    ax.set_xlim(xlim)


def plotPhotErrModelFit(mag, mmag_err, photomModel, color='red', ax=None,
                        verbose=True):
    """Plot model of photometric error from LSST Overview paper (Eq. 4 & 5)

    Parameters
    ----------
    mag : list or numpy.array
        Magnitude
    mmag_err : list or numpy.array
        Magnitude uncertainty or variation in *mmag*.
    photomModel : `AnalyticPhotometryModel`
        Fit parameters to display.
    ax : matplotlib.Axis, optional
        The Axis object to plot to.
    verbose : bool, optional
        Produce extra output to STDOUT
    """

    if ax is None:
        ax = plt.figure()
        xlim = [10, 30]
    else:
        xlim = ax.get_xlim()

    x_model = np.linspace(*xlim, num=100)
    fit_model_mag_err = photErrModel(x_model,
                                     sigmaSys=photomModel.sigmaSys.to(u.mag).value,
                                     gamma=photomModel.gamma.value,
                                     m5=photomModel.m5.to(u.mag).value)
    fit_model_mag_err = fit_model_mag_err * u.mag
    ax.plot(x_model, fit_model_mag_err.to(u.mmag).value,
            color=color, linewidth=2,
            label='Model')

    labelFormatStr = '\n'.join([
        r'$\sigma_\mathrm{{sys}}$ = {sigmaSysMmag:.4f} mmag',
        r'$\gamma = {gamma:.4f}$',
        r'$m_5 =$ {m5:.4f}'])
    label = labelFormatStr.format(sigmaSysMmag=1000*photomModel.sigmaSys.to(u.mag).value,
                                  gamma=photomModel.gamma,
                                  m5=photomModel.m5)
    ax.text(0.1, 0.8, label, color=color,
            transform=ax.transAxes, ha='left', va='top')


def plotPhotometryErrorModel(dataset, photomModel,
                             filterName='', outputPrefix=''):
    """Plot photometric RMS for matched sources.

    Parameters
    ----------
    dataset : `MatchedMultiVisitDataset`
        Blob with the multi-visit photometry model.
    photomModel : `AnalyticPhotometryModel`
        An analyticPhotometry model object.
    filterName : str, optional
        Name of the observed filter to use on axis labels.
    outputPrefix : str, optional
        Prefix to use for filename of plot file.  Will also be used in plot
        titles. E.g., ``outputPrefix='Cfht_output_r_'`` will result in a file
        named ``'Cfht_output_r_check_photometry.png'``.
    """
    bright, = np.where(dataset.snr > photomModel.brightSnr)

    numMatched = len(dataset.mag)
    mmagRms = dataset.magrms.to(u.mmag)
    mmagRmsHighSnr = mmagRms[bright]
    mmagErr = dataset.magerr.to(u.mmag)
    mmagErrHighSnr = mmagErr[bright]

    mmagrms_median = np.median(mmagRms)
    bright_mmagrms_median = np.median(mmagRmsHighSnr)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(18, 16))

    ax[0][0].hist(mmagRms,
                  bins=100, range=(0, 500), color=color['all'],
                  histtype='stepfilled', orientation='horizontal')
    ax[0][0].hist(mmagRmsHighSnr,
                  bins=100, range=(0, 500),
                  color=color['bright'],
                  histtype='stepfilled', orientation='horizontal')
    plotOutlinedAxline(
        ax[0][0].axhline,
        mmagrms_median.value,
        color=color['all'],
        label="Median RMS: {v:.1f}".format(v=mmagrms_median))
    plotOutlinedAxline(
        ax[0][0].axhline,
        bright_mmagrms_median.value,
        color=color['bright'],
        label="SNR > {snr:.0f}\nMedian RMS: {v:.1f}".format(
            snr=photomModel.brightSnr,
            v=bright_mmagrms_median))

    ax[0][0].set_ylim([0, 500])
    ax[0][0].set_ylabel("{magrms.label} [{mmagrms.unit:latex}]".format(
        magrms=dataset.datums['magrms'], mmagrms=mmagRms))
    ax[0][0].legend(loc='upper right')

    ax[0][1].scatter(dataset.mag, mmagRms,
                     s=10, color=color['all'], label='All')
    ax[0][1].scatter(dataset.mag[bright], mmagRmsHighSnr,
                     s=10, color=color['bright'],
                     label='{label} > {value:.0f}'.format(
                         label=photomModel.datums['brightSnr'].label,
                         value=photomModel.brightSnr))
    ax[0][1].set_xlabel("{label} [{unit:latex}]".format(label=filterName,
                                                        unit=dataset.mag.unit))
    ax[0][1].set_ylabel("{label} [{unit:latex}]".format(label=dataset.datums['magrms'].label,
                                                        unit=mmagRmsHighSnr.unit))
    ax[0][1].set_xlim([17, 24])
    ax[0][1].set_ylim([0, 500])
    ax[0][1].legend(loc='upper left')
    plotOutlinedAxline(
        ax[0][1].axhline,
        mmagrms_median.value,
        color=color['all'])
    plotOutlinedAxline(
        ax[0][1].axhline,
        bright_mmagrms_median.value,
        color=color['bright'])
    matchCountTemplate = '\n'.join([
        'Matches:',
        '{nBright:d} high SNR,',
        '{nAll:d} total'])
    ax[0][1].text(0.1, 0.6, matchCountTemplate.format(nBright=len(bright),
                                                      nAll=numMatched),
                  transform=ax[0][1].transAxes, ha='left', va='top')

    ax[1][0].scatter(mmagRms, mmagErr,
                     s=10, color=color['all'], label=None)
    ax[1][0].scatter(mmagRmsHighSnr, mmagErrHighSnr,
                     s=10, color=color['bright'],
                     label=None)
    ax[1][0].set_xscale('log')
    ax[1][0].set_yscale('log')
    ax[1][0].plot([0, 1000], [0, 1000],
                  linestyle='--', color='black', linewidth=2)
    ax[1][0].set_xlabel("{label} [{unit:latex}]".format(
        label=dataset.datums['magrms'].label,
        unit=mmagRms.unit))
    ax[1][0].set_ylabel("Median Reported Magnitude Err [{unit:latex}]".format(
        unit=mmagErr.unit))

    brightSnrMag = 2.5*np.log10(1 + (1/photomModel.brightSnr.value)) * u.mag
    label = r'$SNR > {snr:.0f} \equiv \sigma <  {snrMag:0.1f}$'.format(
        snr=photomModel.brightSnr,
        snrMag=brightSnrMag.to(u.mmag))
    ax[1][0].axhline(brightSnrMag.to(u.mmag).value,
                     color='red', linewidth=4,
                     linestyle='dashed',
                     label=label)
    ax[1][0].set_xlim([1, 500])
    ax[1][0].set_ylim([1, 500])
    ax[1][0].legend(loc='upper center')

    ax[1][1].scatter(dataset.mag, mmagErr,
                     color=color['all'], label=None)
    ax[1][1].set_yscale('log')
    ax[1][1].scatter(np.asarray(dataset.mag)[bright],
                     mmagErrHighSnr,
                     s=10, color=color['bright'],
                     label=None)
    ax[1][1].set_xlabel("{name} [{unit:latex}]".format(
        name=filterName, unit=dataset.mag.unit))
    ax[1][1].set_ylabel("Median Reported Magnitude Err [{unit:latex}]".format(
        unit=mmagErr.unit))
    ax[1][1].set_xlim([17, 24])
    ax[1][1].set_ylim([1, 500])
    ax[1][1].axhline(brightSnrMag.to(u.mmag).value,
                     color='red', linewidth=4,
                     linestyle='dashed',
                     label=None)

    w, = np.where(mmagErr < 200. * u.mmag)
    plotPhotErrModelFit(dataset.mag[w].to(u.mag).value,
                        dataset.magerr[w].to(u.mmag).value,
                        photomModel, ax=ax[1][1])
    ax[1][1].legend(loc='upper left')

    plt.suptitle("Photometry Check : %s" % outputPrefix,
                 fontsize=30)
    plotPath = outputPrefix+"check_photometry.png"
    plt.savefig(plotPath, format="png")
    plt.close(fig)


def plotPA1(pa1, outputPrefix=""):
    """Plot the results of calculating the LSST SRC requirement PA1.

    Creates a file containing the plot with a filename beginning with
    `outputPrefix`.

    Parameters
    ----------
    pa1 : `PA1Measurement`
        A PA1 object.
    outputPrefix : `str`, optional
        Prefix to use for filename of plot file.  Will also be used in plot
        titles. E.g., outputPrefix='Cfht_output_r_' will result in a file
        named ``'Cfht_output_r_AM1_D_5_arcmin_17.0-21.5.png'``
        for an ``AMx.name=='AM1'`` and ``AMx.magRange==[17, 21.5]``.
    """
    diffRange = (-100, +100)

    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(pa1.magMean[0],
                pa1.magDiff[0],
                s=10, color=color['bright'], linewidth=0)
    # index 0 because we show only the first sample from multiple trials
    ax1.axhline(+pa1.rms[0].value, color=color['rms'], linewidth=3)
    ax1.axhline(-pa1.rms[0].value, color=color['rms'], linewidth=3)
    ax1.axhline(+pa1.iqr[0].value, color=color['iqr'], linewidth=3)
    ax1.axhline(-pa1.iqr[0].value, color=color['iqr'], linewidth=3)

    ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
    ax2.hist(pa1.magDiff[0], bins=25, range=diffRange,
             orientation='horizontal', histtype='stepfilled',
             normed=True, color=color['bright'])
    ax2.set_xlabel("relative # / bin")

    labelTemplate = r'PA1({label}) = {q.value:4.2f} {q.unit:latex}'
    yv = np.linspace(diffRange[0], diffRange[1], 100)
    ax2.plot(scipy.stats.norm.pdf(yv, scale=pa1.rms[0]), yv,
             marker='', linestyle='-', linewidth=3, color=color['rms'],
             label=labelTemplate.format(label='RMS', q=pa1.rms[0]))
    ax2.plot(scipy.stats.norm.pdf(yv, scale=pa1.iqr[0]), yv,
             marker='', linestyle='-', linewidth=3, color=color['iqr'],
             label=labelTemplate.format(label='IQR', q=pa1.iqr[0]))
    ax2.set_ylim(*diffRange)
    ax2.legend()
    ax1.set_xlabel("psf magnitude")
    ax1.set_ylabel(r"psf magnitude diff ({0.unit:latex})".format(pa1.extras['magDiff'].quantity))
    for label in ax2.get_yticklabels():
        label.set_visible(False)

    plt.tight_layout()  # fix padding
    if outputPrefix == '':
        plotPath = "PA1.png"
    else:
        plotPath = "%s_%s" % (outputPrefix, "PA1.png")
    plt.savefig(plotPath, format="png")
    plt.close(fig)


def plotAMx(amx, afx, filterName, amxSpecName='design', outputPrefix=""):
    """Plot a histogram of the RMS in relative distance between pairs of
    stars.

    Creates a file containing the plot with a filename beginning with
    `outputPrefix`.

    Parameters
    ----------
    amx : `AMxMeasurement`
    afx : `AFxMeasurement`
    amxSpecName : `str`, optional
        Name of the AMx specification to reference in the plot.
        Default: ``'design'``.
    outputPrefix : `str`, optional
        Prefix to use for filename of plot file.  Will also be used in plot
        titles. E.g., ``outputPrefix='Cfht_output_r_'`` will result in a file
        named ``'Cfht_output_r_AM1_D_5_arcmin_17.0-21.5.png'``
        for an ``AMx.name=='AM1'`` and ``AMx.magRange==[17, 21.5]``.
    """
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)

    histLabelTemplate = 'D: [{inner.value:.1f}{inner.unit:latex}-{outer.value:.1f}{outer.unit:latex}]\n'\
                        'Mag: [{magBright:.1f}-{magFaint:.1f}]'
    ax1.hist(amx.rmsDistMas, bins=25, range=(0.0, 100.0),
             histtype='stepfilled',
             label=histLabelTemplate.format(
                 inner=amx.annulus[0],
                 outer=amx.annulus[1],
                 magBright=amx.magRange[0],
                 magFaint=amx.magRange[1]))

    amxSpec = amx.metric.get_spec(amxSpecName, filter_name=filterName)
    amxSpecLabelTemplate = '{amx.label} {specname}: {amxSpec.quantity:.1f}'
    amxSpecLabel = amxSpecLabelTemplate.format(
        amx=amx,
        specname=amxSpecName,
        amxSpec=amxSpec)
    ax1.axvline(amxSpec.quantity.value, 0, 1, linewidth=2, color='red',
                label=amxSpecLabel)

    if amx.check_spec(amxSpecName):
        amxStatus = 'passed'
    else:
        amxStatus = 'failed'
    amxLabelTemplate = '{amx.label} measured: {amx.quantity:.1f} ({status})'
    amxLabel = amxLabelTemplate.format(amx=amx, status=amxStatus)
    ax1.axvline(amx.quantity.value, 0, 1, linewidth=2, color='black',
                label=amxLabel)

    if afx.check_spec(afx.spec_name):
        afxStatus = 'passed'
    else:
        afxStatus = 'failed'
    afxSpec = afx.metric.get_spec(afx.spec_name, filter_name=filterName)
    afxLabelTemplate = '{afx.label} {afx.spec_name}: {afxSpec.quantity}%\n' + \
                       '{afx.label} measured: {afx.quantity:.1f}% ({status})'
    afxLabel = afxLabelTemplate.format(
        afx=afx,
        afxSpec=afxSpec,
        status=afxStatus)
    ax1.axvline((amx.quantity + afx.ADx).value,
                0, 1, linewidth=2, color='green',
                label=afxLabel)

    title = '{metric} Astrometric Repeatability over {D.value:.0f}{D.unit:latex}'.format(
        metric=amx.label,
        D=amx.D)
    ax1.set_title(title)
    ax1.set_xlim(0.0, 100.0)
    ax1.set_xlabel(
        '{rmsDistMas.label} ({rmsDistMas.latex_unit})'.format(
            rmsDistMas=amx.extras['rmsDistMas']))
    ax1.set_ylabel('# pairs / bin')

    ax1.legend(loc='upper right', fontsize=16)

    pathFormat = '{prefix}{metric}_D_{D:d}_{Dunits}_' + \
                 '{magBright.value}_{magFaint.value}_{magFaint.unit}.{ext}'
    plotPath = pathFormat.format(
        prefix=outputPrefix,
        metric=amx.label,
        D=int(amx.D.value),
        Dunits=amx.parameters['D'].unit,
        magBright=amx.magRange[0],
        magFaint=amx.magRange[1],
        ext='png')

    plt.tight_layout()  # fix padding
    plt.savefig(plotPath, dpi=300)
    plt.close(fig)


def plotTEx(tex, filterName, texSpecName='design', outputPrefix=''):
    """Plot TEx correlation function measurements and thresholds.

    Parameters
    ----------
    tex : Measurement object
        The ellipticity residual correlation Measurement object
    filterName : str
        Name of the filter of the images
    texSpecName : str
        Level of requirement to compare against.
        Must be a into the metrics specified in the tex Measurement object
        Typically one of 'design', 'minimum', 'stretch'
    outputPrefix : str, optional
        Prefix to use for filename of plot file.

    Effects
    -------
    Saves an output plot file to that starts with specified outputPrefix.

    """
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    # Plot correlation vs. radius
    ax1.errorbar(tex.radius.value, tex.xip.value, yerr=tex.xip_err.value)
    ax1.set_xscale('log')
    ax1.set_xlabel('Separation (arcmin)', size=19)
    ax1.set_ylabel('Median Residual Ellipticity Correlation', size=19)

    # Overlay requirements level
    texSpec = tex.metric.get_spec(texSpecName, filter_name=filterName)
    texSpecLabel = '{tex.label} {specname}: {texSpec:.2g}'.format(
        tex=tex,
        texSpec=tex.metric.get_spec(texSpecName,
                                    filter_name=filterName).quantity,
        specname=texSpecName)
    ax1.axhline(texSpec.quantity.value, 0, 1, linewidth=2, color='red',
                label=texSpecLabel)

    # Overlay measured KPM whether it passed or failed.
    if tex.check_spec(texSpecName):
        texStatus = 'passed'
    else:
        texStatus = 'failed'
    texLabelTemplate = '{tex.label} measured: {tex.quantity:.2g} ({status})'
    texLabel = texLabelTemplate.format(tex=tex, status=texStatus)

    ax1.axhline(tex.quantity.value, 0, 1, linewidth=2, color='black',
                label=texLabel)

    titleTemplate = """
        {metric} Residual PSF Ellipticity Correlation
        {bin_range_operator:s} {D.value:.1f}{D.unit:latex}
        """
    title = titleTemplate.format(metric=tex.label,
                                 bin_range_operator=tex.bin_range_operator,
                                 D=tex.D)
    ax1.set_title(title)
    ax1.set_xlim(0.0, 20.0)
    ax1.set_xlabel(
        '{radius.label} ({radius.latex_unit})'.format(
            radius=tex.extras['radius']))
    ax1.set_ylabel('Correlation')

    ax1.legend(loc='upper right', fontsize=16)

    pathFormat = '{prefix}{metric}_D_{D:d}_{Dunits}.{ext}'
    plotPath = pathFormat.format(
        prefix=outputPrefix,
        metric=tex.label,
        D=int(tex.D.value),
        Dunits=tex.parameters['D'].unit,
        ext='png')

    plt.tight_layout()  # fix padding
    plt.savefig(plotPath, dpi=300)
    plt.close(fig)
