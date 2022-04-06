""" Color modes for graphed pintk TOAs. """
import logging

import numpy as np
import matplotlib

log = logging.getLogger(__name__)


class ColorMode:
    """ Base Class for color modes. """

    def __init__(self, application):
        self.application = application  # PLKWidget for pintk

    def displayInfo(self):
        raise NotImplementedError

    def plotColorMode(self):
        raise NotImplementedError


class DefaultMode(ColorMode):
    """
    A class to manage the Default color mode, where TOAs are colored
    blue as a default and red if jumped.
    """

    def __init__(self, application):
        super(DefaultMode, self).__init__(application)
        self.mode_name = "default"

    def displayInfo(self):
        log.info(
            '"Default" mode selected\nBlue = default color\nOrange = '
            + "selected TOAs\nRed = jumped TOAs"
        )

    def plotColorMode(self):
        """
        Plots application's residuals in proper color scheme.
        """
        if self.application.yerrs is None:
            self.application.plkAxes.scatter(
                self.application.xvals[~self.application.selected],
                self.application.yvals[~self.application.selected],
                marker=".",
                color="blue",
            )
            self.application.plkAxes.scatter(
                self.application.xvals[self.application.jumped],
                self.application.yvals[self.application.jumped],
                marker=".",
                color="red",
            )
            self.application.plkAxes.scatter(
                self.application.xvals[self.application.selected],
                self.application.yvals[self.application.selected],
                marker=".",
                color="orange",
            )
        else:
            self.application.plotErrorbar(~self.application.selected, color="blue")
            self.application.plotErrorbar(self.application.jumped, color="red")
            self.application.plotErrorbar(self.application.selected, color="orange")


class FreqMode(ColorMode):
    """
    A class to manage the Frequency color mode, where TOAs are colored
    according to their frequency.
    """

    def __init__(self, application):
        super(FreqMode, self).__init__(application)
        self.mode_name = "freq"

    def displayInfo(self):
        log.warning(
            '"Frequency" mode selected\n'
            + "  Dark Red <  300 MHz\n"
            + "  Red      =  300-400  MHz\n"
            + "  Orange   =  400-500  MHz\n"
            + "  Yellow   =  500-700  MHz\n"
            + "  Green    =  700-1000 MHz\n"
            + "  Blue     = 1000-1800 MHz\n"
            + "  Indigo   = 1800-3000 MHz\n"
            + "  Black    = 3000-8000 MHz\n"
            + "  Grey     > 8000 MHz\n"
            + "  Brown is for selected TOAs\n"
        )

    def plotColorMode(self):
        """
        Plots application's residuals in proper color scheme.
        """

        colorGroups = [
            "#8C0000",  # dark red
            "#FF0000",  # red
            "#FFA500",  # orange
            "#FFF133",  # yellow-ish
            "#008000",  # green
            "#0000FF",  # blue
            "#4B0082",  # indigo
            "#000000",  # black
            "#7E7E7E",  # grey
        ]
        highfreqs = [300.0, 400.0, 500.0, 700.0, 1000.0, 1800.0, 3000.0, 8000.0]

        freqGroups = []
        for ii, highfreq in enumerate(highfreqs):
            if ii == 0:
                freqGroups.append(
                    np.where(self.application.psr.all_toas.get_freqs().value < highfreq)
                )
            else:
                freqGroups.append(
                    np.where(
                        (self.application.psr.all_toas.get_freqs().value < highfreq)
                        & (
                            self.application.psr.all_toas.get_freqs().value
                            >= highfreqs[ii - 1]
                        )
                    )
                )
        freqGroups.append(
            np.where(self.application.psr.all_toas.get_freqs().value >= highfreqs[-1])
        )

        for index in range(len(freqGroups)):
            if self.application.yerrs is None:
                self.application.plkAxes.scatter(
                    self.application.xvals[freqGroups[index]],
                    self.application.yvals[freqGroups[index]],
                    marker=".",
                    color=colorGroups[index],
                )
            else:
                self.application.plotErrorbar(
                    freqGroups[index], color=colorGroups[index]
                )
        # The following is for selected TOAs
        if self.application.yerrs is None:
            self.application.plkAxes.scatter(
                self.application.xvals[self.application.selected],
                self.application.yvals[self.application.selected],
                marker=".",
                color="#362511",  # brown
            )
        else:
            self.application.plotErrorbar(self.application.selected, color="#362511")


class NameMode(ColorMode):
    """
    A class to manage the Frequency color mode, where TOAs are colored
    according to their frequency.
    """

    def __init__(self, application):
        super(NameMode, self).__init__(application)
        self.mode_name = "name"

    def displayInfo(self):
        log.info('"Name" mode selected\nOrange = selected TOAs')

    def plotColorMode(self):
        """
        Plots application's residuals in proper color scheme.
        """

        all_names = np.array(
            [f["name"] for f in self.application.psr.all_toas.get_flags()]
        )
        single_names = list(set(all_names))
        N = len(single_names)
        cmap = matplotlib.cm.get_cmap("brg")
        colorGroups = [matplotlib.colors.rgb2hex(cmap(v)) for v in np.linspace(0, 1, N)]
        colorGroups += ["orange"]

        freqGroups = []
        index = 0
        for name in single_names:
            index += 1
            freqGroups.append(np.where(all_names == name))
            index += 1

        for index in range(N):
            if self.application.yerrs is None:
                self.application.plkAxes.scatter(
                    self.application.xvals[freqGroups[index]],
                    self.application.yvals[freqGroups[index]],
                    marker=".",
                    color=colorGroups[index],
                )
            else:
                self.application.plotErrorbar(
                    freqGroups[index], color=colorGroups[index]
                )

        if self.application.yerrs is None:
            self.application.plkAxes.scatter(
                self.application.xvals[self.application.selected],
                self.application.yvals[self.application.selected],
                marker=".",
                color=colorGroups[N],
            )
        else:
            self.application.plotErrorbar(
                self.application.selected, color=colorGroups[N]
            )


class ObsMode(ColorMode):
    """
    A class to manage the Observatory color mode, where TOAs are colored
    according to their observatory.
    """

    def __init__(self, application):
        super(ObsMode, self).__init__(application)
        self.mode_name = "obs"

    def displayInfo(self):
        log.info(
            '"Observatory" mode selected\nBlue = barycenter, gmrt, and virgo'
            + "\nRed = geocenter, wsrt, and lho\nGray = spacecenter, fast, and llo\n"
            + "Cyan = gbt, mwa, and geo600\nLight Green = arecibo, lwa1, and kagra\n"
            + "Burnt Orange = vla, ps1, and algonquin\nPink = parkes, hobart, and drao"
            + "\nDark Green = jodrell, most, and acre\nMaroon = nancay and chime\n"
            + "Sky Blue = ncyobs and magic\nLight Purple = effelsberg and lst\nOrange"
            + " = selected TOAs"
        )

    def plotColorMode(self):
        """
        Plots application's residuals in proper color scheme.
        """
        import pint.observatory as obs
        import pint.observatory.topo_obs, pint.observatory.observatories

        observatories = obs.Observatory.names()
        colorGroups = [
            "blue",
            "red",
            "gray",
            "cyan",
            "#00FF33",  # light green
            "#CC6600",  # burnt orange
            "#FF00FF",  # pink
            "#006600",  # dark green
            "#990000",  # maroon
            "#0099FF",  # sky blue
            "#6699FF",  # light purple
            "orange",
        ]

        obsGroups = []
        index = 0
        for observatory in observatories:
            # group toa indeces by observatory
            obsGroups.append(
                np.where(self.application.psr.all_toas.get_obss() == observatory)
            )
            index += 1

        cindex = 0
        for i in range(index):
            if cindex == 11:  # max number of colors currently available
                cindex = 0
            if self.application.yerrs is None:
                self.application.plkAxes.scatter(
                    self.application.xvals[obsGroups[i]],
                    self.application.yvals[obsGroups[i]],
                    marker=".",
                    color=colorGroups[cindex],
                )
            else:
                self.application.plotErrorbar(obsGroups[i], color=colorGroups[cindex])
            i += 1
            cindex += 1
        if self.application.yerrs is None:
            self.application.plkAxes.scatter(
                self.application.xvals[self.application.selected],
                self.application.yvals[self.application.selected],
                marker=".",
                color=colorGroups[11],
            )
        else:
            self.application.plotErrorbar(
                self.application.selected, color=colorGroups[11]
            )
