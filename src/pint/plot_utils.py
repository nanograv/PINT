#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import astropy
import astropy.units as u
import astropy.time
from loguru import logger as log

__all__ = ["phaseogram", "phaseogram_binned", "plot_priors"]


def phaseogram(
    mjds_in,
    phases,
    weights=None,
    title=None,
    bins=100,
    rotate=0.0,
    size=5,
    alpha=0.25,
    width=6,
    maxphs=2.0,
    plotfile=None,
):
    """Make a nice 2-panel phaseogram

    Makes a phaseogram of photons with phases, with a point for each photon that can
    have a transparency determined by an array of weights.

    Parameters
    ----------
    mjds_in : array or astropy.units.Quantity or astropy.time.Time
        Assumes units of days if bare numerical array, otherwise
        will convert Quantity or Time into days.
    phases : array
        Phases for each photon, assumes range is [0,1)

    """
    # If mjds_in is a Time() then pull out the MJD values and make a quantity
    if type(mjds_in) == astropy.time.core.Time:
        mjds = mjds_in.mjd * u.d
    # If mjds_in have no units, assume days
    elif type(mjds_in) != astropy.units.quantity.Quantity:
        mjds = mjds_in * u.d
    else:
        mjds = mjds_in

    years = (mjds.to(u.d).value - 51544.0) / 365.25 + 2000.0
    phss = phases + rotate
    phss[phss >= 1.0] -= 1.0
    plt.figure(figsize=(width, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
    wgts = None if weights is None else np.concatenate((weights, weights))
    h, x, p = ax1.hist(
        np.concatenate((phss, phss + 1.0)),
        int(maxphs * bins),
        range=[0, maxphs],
        weights=wgts,
        color="k",
        histtype="step",
        fill=False,
        lw=2,
    )
    ax1.set_xlim([0.0, maxphs])  # show 1 or more pulses
    ax1.set_ylim([0.0, 1.1 * h.max()])
    if weights is not None:
        ax1.set_ylabel("Weighted Counts")
    else:
        ax1.set_ylabel("Counts")
    if title is not None:
        ax1.set_title(title)
    if weights is None:
        ax2.scatter(phss, mjds, s=size, color="k", alpha=alpha)
        ax2.scatter(phss + 1.0, mjds, s=size, color="k", alpha=alpha)
    else:
        colarray = np.array([[0.0, 0.0, 0.0, w] for w in weights])
        ax2.scatter(phss, mjds, s=size, color=colarray)
        ax2.scatter(phss + 1.0, mjds, s=size, color=colarray)
    ax2.set_xlim([0.0, maxphs])  # show 1 or more pulses
    ax2.set_ylim([mjds.min().value, mjds.max().value])
    ax2.set_ylabel("MJD")
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.get_yaxis().get_major_formatter().set_scientific(False)
    ax2r = ax2.twinx()
    ax2r.set_ylim([years.min(), years.max()])
    ax2r.set_ylabel("Year")
    ax2r.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2r.get_yaxis().get_major_formatter().set_scientific(False)
    ax2.set_xlabel("Pulse Phase")
    plt.tight_layout()
    if plotfile is not None:
        plt.savefig(plotfile)
        plt.close()
    else:
        plt.show()


def phaseogram_binned(
    mjds_in,
    phases,
    weights=None,
    title=None,
    bins=64,
    rotate=0.0,
    size=5,
    alpha=0.25,
    width=6,
    maxphs=2.0,
    plotfile=None,
):
    """
    Make a nice 2-panel phaseogram

    Makes a binned phaseogram of photons with phases, where the contribution to each bin
    can be determined by an array of weights.

    Parameters
    ----------
    mjds_in : array or astropy.units.Quantity or astropy.time.Time
        Assumes units of days if bare numerical array, otherwise
        will convert Quantity or Time into days.
    phases : array
        Phases for each photon, assumes range is [0,1)

    """
    # If mjds_in is a Time() then pull out the MJD values and make a quantity
    if type(mjds_in) == astropy.time.core.Time:
        mjds = mjds_in.mjd * u.d
    # If mjds_in has no units, assume days
    elif type(mjds_in) != astropy.units.quantity.Quantity:
        mjds = mjds_in * u.d
    else:
        mjds = mjds_in

    years = (mjds.to(u.d).value - 51544.0) / 365.25 + 2000.0
    phss = phases + rotate
    phss[phss >= 1.0] -= 1.0
    plt.figure(figsize=(width, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
    wgts = None if weights is None else np.concatenate((weights, weights))
    h, x, p = ax1.hist(
        np.concatenate((phss, phss + 1.0)),
        int(maxphs * bins),
        range=[0, maxphs],
        weights=wgts,
        color="k",
        histtype="step",
        fill=False,
        lw=2,
    )
    ax1.set_xlim([0.0, maxphs])  # show 1 or more pulses
    ax1.set_ylim([0.0, 1.1 * h.max()])
    if weights is not None:
        ax1.set_ylabel("Weighted Counts")
    else:
        ax1.set_ylabel("Counts")
    if title is not None:
        ax1.set_title(title)
    SCATTER = False
    if SCATTER:
        if weights is None:
            ax2.scatter(phss, mjds, s=size, color="k", alpha=alpha)
            ax2.scatter(phss + 1.0, mjds, s=size, color="k", alpha=alpha)
        else:
            colarray = np.array([[0.0, 0.0, 0.0, w] for w in weights])
            ax2.scatter(phss, mjds, s=size, color=colarray)
            ax2.scatter(phss + 1.0, mjds, s=size, color=colarray)
    else:
        profile = np.zeros(bins, dtype=np.float_)
        ntoa = 64
        toadur = (mjds.max() - mjds.min()) / ntoa
        mjdstarts = mjds.min() + toadur * np.arange(ntoa, dtype=np.float_)
        mjdstops = mjdstarts + toadur
        # Loop over blocks to process
        a = []
        for tstart, tstop in zip(mjdstarts, mjdstops):
            # Clear profile array
            profile = profile * 0.0

            idx = (mjds > tstart) & (mjds < tstop)

            if weights is not None:
                for ph, ww in zip(phases[idx], weights[idx]):
                    ibin = int(ph * bins)
                    profile[ibin] += ww
            else:
                for ph in phases[idx]:
                    ibin = int(ph * bins)
                    profile[ibin] += 1

            a.extend(profile[i] for i in range(bins))

        a = np.array(a)
        b = a.reshape(ntoa, bins)
        c = np.hstack([b, b])
        ax2.imshow(
            c,
            interpolation="nearest",
            origin="lower",
            cmap=plt.cm.binary,
            extent=[0, 2.0, mjds.min().value, mjds.max().value],
            aspect="auto",
        )

    ax2.set_xlim([0.0, maxphs])  # show 1 or more pulses
    ax2.set_ylim([mjds.min().value, mjds.max().value])
    ax2.set_ylabel("MJD")
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.get_yaxis().get_major_formatter().set_scientific(False)
    ax2r = ax2.twinx()
    ax2r.set_ylim([years.min(), years.max()])
    ax2r.set_ylabel("Year")
    ax2r.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2r.get_yaxis().get_major_formatter().set_scientific(False)
    ax2.set_xlabel("Pulse Phase")
    plt.tight_layout()
    if plotfile is not None:
        plt.savefig(plotfile)
        plt.close()
    else:
        plt.show()


def plot_priors(
    model,
    chains,
    maxpost_fitvals=None,
    fitvals=None,
    burnin=100,
    bins=100,
    scale=False,
):
    """Plot of priors and the post-MCMC histogrammed samples

    Show binned samples, prior probability distribution and an initial
    gaussian probability distribution plotted with 2 sigma, maximum
    posterior and original fit values marked.

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        The initial timing model for fitting
    chains : dict
        Post MCMC integration chains that contain the fitter keys and post
        MCMC samples, which are histogrammed and normalized. Thinning the
        samples from the chains is not supported. Can be created using
        :meth:`pint.sampler.EmceeSampler.chains_to_dict`
    maxpost_fitvals : list, optional
        The maximum posterior values returned from MCMC integration for each
        fitter key. Plots a vertical dashed line to denote the maximum
        posterior value in relation to the histogrammed samples. If the
        values are not provided, then the lines are not plotted
    fitvals : list, optional
        The original parameter fit values. Plots vertical dashed lines to
        denote the original parameter fit values in relation to the
        histogrammed samples. If the values are not provided, then the
        lines are not plotted.
    burnin : int
        The number of steps that are the burnin in the MCMC integration
    bins : int
        Number of bins used in the histogram
    scale : bool
        If True, the priors will be scaled to the peak of the histograms.
        If False, the priors will be plotted independent of the histograms.
        In certain cases, such as broad priors, the priors or histograms
        might not be apparent on the same plot due to one being significantly
        larger than the other. The scaling is for visual purposes to clearly
        plot the priors with the samples
    """
    values = []
    keys = []
    for k, v in chains.items():
        keys.append(k), values.append(v)

    priors = []
    x_range = []
    counts = []
    for i in range(len(keys[:-1])):
        values[i] = values[i][burnin:].flatten()
        x_range.append(np.linspace(values[i].min(), values[i].max(), num=bins))
        priors.append(getattr(model, keys[i]).prior.pdf(x_range[i]))
        a, x = np.histogram(values[i], bins=bins, density=True)
        counts.append(a)

    fig, axs = plt.subplots(len(keys), figsize=(8, 11), constrained_layout=True)

    for i, p in enumerate(keys):
        if i != len(keys[:-1]):
            axs[i].set_xlabel(
                f"{str(p)}: Mean Value = "
                + "{:.9e}".format(values[i].mean())
                + " ("
                + str(getattr(model, p).units)
                + ")"
            )
            axs[i].axvline(
                -2 * values[i].std(), color="b", linestyle="--", label="2 sigma"
            )
            axs[i].axvline(2 * values[i].std(), color="b", linestyle="--")
            axs[i].hist(
                values[i] - values[i].mean(), bins=bins, density=True, label="Samples"
            )
            if scale:
                axs[i].plot(
                    x_range[i] - values[i].mean(),
                    priors[i] * counts[i].max() / priors[i].max(),
                    label="Prior Probability",
                    color="g",
                )
            else:
                axs[i].plot(
                    x_range[i] - values[i].mean(),
                    priors[i],
                    label="Prior Probability",
                    color="g",
                )
            if maxpost_fitvals is not None:
                axs[i].axvline(
                    maxpost_fitvals[i] - values[i].mean(),
                    color="c",
                    linestyle="--",
                    label="Maximum Likelihood Value",
                )
            if fitvals is not None:
                axs[i].axvline(
                    fitvals[i] - values[i].mean(),
                    color="m",
                    linestyle="--",
                    label="Original Parameter Fit Value",
                )
        else:
            handles, labels = axs[0].get_legend_handles_labels()
            axs[i].set_axis_off()
            axs[i].legend(handles, labels)
