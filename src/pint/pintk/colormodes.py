""" Color modes for graphed pintk TOAs. """
import numpy as np
import matplotlib
import matplotlib.colors

import pint.logging
from loguru import logger as log

# subset of other colors to allow users to distinguish between them
named_colors = [
    "xkcd:red",
    "xkcd:green",
    "xkcd:cyan",
    "xkcd:blue",
    "xkcd:burnt orange",
    "xkcd:brown",
    "xkcd:indigo",
    "xkcd:purple",
    "xkcd:dark blue",
    "xkcd:light green",
    "xkcd:dark green",
    "lxkcd:ight blue",
    "xkcd:dark red",
    "xkcd:magenta",
    "xkcd:black",
    "xkcd:grey",
    "xkcd:light grey",
    "xkcd:yellow",
    "xkcd:orange",
]


class ColorMode:
    """Base Class for color modes."""

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
        print(
            '"Default" mode selected\n'
            + "  Blue   = default color\n"
            + "  Orange = selected TOAs\n"
            + "  Red    = jumped TOAs\n"
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
        print(
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
            "xkcd:dark red",  # dark red
            "xkcd:red",  # red
            "xkcd:orange",  # orange
            "xkcd:yellow",  # yellow
            "xkcd:green",  # green
            "xkcd:blue",  # blue
            "xkcd:indigo",  # indigo
            "xkcd:black",  # black
            "xkcd:grey",  # grey
        ]
        highfreqs = [300.0, 400.0, 500.0, 700.0, 1000.0, 1800.0, 3000.0, 8000.0]

        freqGroups = []
        for ii, highfreq in enumerate(highfreqs):
            if ii == 0:
                freqGroups.append(
                    self.application.psr.all_toas.get_freqs().value < highfreq
                )
            else:
                freqGroups.append(
                    (self.application.psr.all_toas.get_freqs().value < highfreq)
                    & (
                        self.application.psr.all_toas.get_freqs().value
                        >= highfreqs[ii - 1]
                    )
                )
        freqGroups.append(
            self.application.psr.all_toas.get_freqs().value >= highfreqs[-1]
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
    according to their names in the TOA lines.
    """

    def __init__(self, application):
        super(NameMode, self).__init__(application)
        self.mode_name = "name"

    def displayInfo(self):
        print('"Name" mode selected\n' + "  Orange = selected TOAs\n")

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
            freqGroups.append(all_names == name)
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

    def get_obs_mapping(self):
        "This maps the obs names in the TOAs to our local subset"
        tmp_obs = self.application.psr.all_toas.observatories
        mapping = {}
        for oo in tmp_obs:
            if "stl" in oo:
                mapping[oo] = "space"
            elif oo.startswith("gb"):
                mapping[oo] = "gb"
            elif oo.startswith("jb"):
                mapping[oo] = "jodrell"
            elif "ncy" in oo:
                mapping[oo] = "nancay"
            else:
                mapping[oo] = oo if oo in self.obs_colors else "other"
        return mapping

    obs_colors = {
        "parkes": "xkcd:red",
        "gb": "xkcd:green",  # this is any green bank obs
        "jodrell": "xkcd:cyan",
        "arecibo": "xkcd:blue",
        "chime": "xkcd:burnt orange",
        "gmrt": "xkcd:brown",
        "vla": "xkcd:indigo",
        "effelsberg": "xkcd:purple",
        "fast": "xkcd:dark blue",
        "nancay": "xkcd:light green",
        "srt": "xkcd:dark green",
        "wsrt": "xkcd:light blue",
        "lofar": "xkcd:dark red",
        "lwa": "xkcd:dark red",
        "mwa": "xkcd:dark red",
        "meerkat": "xkcd:magenta",
        "barycenter": "xkcd:black",
        "geocenter": "xkcd:grey",
        "space": "xkcd:light grey",
        "other": "xkcd:yellow",
    }

    obs_text = {
        "parkes": "  Parkes = red",
        "gb": "  Green Bank = green",
        "jodrell": "  Jodrell = cyan",
        "arecibo": "  Arecibo = blue",
        "chime": "  CHIME = burnt orange",
        "gmrt": "  GMRT = brown",
        "vla": "  VLA = indigo",
        "effelsberg": "  Effelsberg = purple",
        "fast": "  FAST = dark blue",
        "nancay": "  Nancay = light green",
        "srt": "  SRT = dark green",
        "wsrt": "  WSRT = light blue",
        "lofar": "  LOFAR = dark red",
        "lwa": "  LWA = dark red",
        "mwa": "  MWA = dark red",
        "meerkat": "  MeerKAT = magenta",
        "barycenter": "  barycenter = black",
        "geocenter": "  geocenter = grey",
        "space": "  satellite = light grey",
        "other": "  other = yellow",
    }

    def displayInfo(self):
        outstr = '"Observatory" mode selected\n'
        for obs in self.get_obs_mapping().values():
            outstr += self.obs_text[obs].replace("xkcd", "") + "\n"
        outstr += "  selected = orange\n"
        print(outstr)

    def plotColorMode(self):
        """
        Plots application's residuals in proper color scheme.
        """
        obsmap = self.get_obs_mapping()
        alltoas = self.application.psr.all_toas
        for obs, ourobs in obsmap.items():
            # group toa indices by observatory
            toas = alltoas.get_obss() == obs
            color = self.obs_colors[ourobs]
            if self.application.yerrs is None:
                self.application.plkAxes.scatter(
                    self.application.xvals[toas],
                    self.application.yvals[toas],
                    marker=".",
                    color=color,
                )
            else:
                self.application.plotErrorbar(toas, color=color)
        # Now handle the selected TOAs
        if self.application.yerrs is None:
            self.application.plkAxes.scatter(
                self.application.xvals[self.application.selected],
                self.application.yvals[self.application.selected],
                marker=".",
                color="orange",
            )
        else:
            self.application.plotErrorbar(self.application.selected, color="orange")


class JumpMode(ColorMode):
    """Mode to color points according to jump"""

    def __init__(self, application):
        super(JumpMode, self).__init__(application)
        self.mode_name = "jump"

    def get_jumps(self):
        if self.application.psr.fitted:
            model = self.application.psr.postfit_model
        else:
            model = self.application.psr.prefit_model
        return model.get_jump_param_objects()

    jump_colors = named_colors
    selected_color = "xkcd:orange"

    def displayInfo(self):
        outstr = '"Jump" mode selected\n'
        for jumpnum, jump in enumerate(self.get_jumps()):
            # only use the number of colors - 1 to preserve orange for selected
            color_number = jumpnum % (len(self.jump_colors) - 1)
            color_name = self.jump_colors[color_number]
            outstr += f"{jump.name}"
            if jump.key is not None:
                outstr += f" {jump.key}"
            if jump.key_value is not None:
                outstr += " " + " ".join(jump.key_value)
            outstr += f" = {color_name.replace('xkcd:','')}\n"
        outstr += f"  selected = {self.selected_color}\n"
        print(outstr)

    def plotColorMode(self):
        """Plot the points with the desired coloring"""
        alltoas = self.application.psr.all_toas
        for jumpnum, jump in enumerate(self.get_jumps()):
            color_number = jumpnum % (len(self.jump_colors) - 1)
            color_name = self.jump_colors[color_number]
            toas = jump.select_toa_mask(alltoas)
            # group toa indices by jump
            if self.application.yerrs is None:
                self.application.plkAxes.scatter(
                    self.application.xvals[toas],
                    self.application.yvals[toas],
                    marker=".",
                    color=color_name,
                )
            else:
                self.application.plotErrorbar(toas, color=color_name)
        # Now handle the selected TOAs
        if self.application.yerrs is None:
            self.application.plkAxes.scatter(
                self.application.xvals[self.application.selected],
                self.application.yvals[self.application.selected],
                marker=".",
                color=self.selected_color,
            )
        else:
            self.application.plotErrorbar(
                self.application.selected, color=self.selected_color
            )
