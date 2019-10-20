#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["phaseogram", "phaseogram_binned"]


def phaseogram(
    mjds,
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
    """Make a nice 2-panel phaseogram"""
    years = (mjds.value - 51544.0) / 365.25 + 2000.0
    phss = phases + rotate
    phss[phss > 1.0] -= 1.0
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
    mjds,
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
    """
    years = (mjds.value - 51544.0) / 365.25 + 2000.0
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
                    bin = int(ph * bins)
                    profile[bin] += ww
            else:
                for ph in phases[idx]:
                    bin = int(ph * bins)
                    profile[bin] += 1

            for i in range(bins):
                a.append(profile[i])

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
