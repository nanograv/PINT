from pint.models import TimingModel
from pint.models.noise_model import NoiseComponent
from pint.models.parameter import (
    AngleParameter,
    MJDParameter,
    boolParameter,
    floatParameter,
    intParameter,
    maskParameter,
    prefixParameter,
    strParameter,
)
from pint.toa import TOAs
from pint.residuals import Residuals
from io import StringIO
import numpy as np
from uncertainties import ufloat


def publish(
    model: TimingModel,
    toas: TOAs,
    include_dmx=False,
    include_noise=False,
    include_jumps=True,
    include_zeros=False,
):
    psrname = model.PSR.value
    nfree = len(model.free_params)
    ephem = model.EPHEM.value
    clock = model.CLOCK.value
    timeeph = model.TIMEEPH.value
    units = model.UNITS.value

    mjds = toas.get_mjds()
    mjd_start, mjd_end = int(min(mjds.value)), int(max(mjds.value))
    data_span_yr = (mjd_end - mjd_start) / 365.25
    ntoas = len(toas)

    toa_type = "Wideband" if toas.is_wideband() else "Narrowband"
    fit_method = (
        "GLS"
        if np.any([nc.introduces_correlated_errors for nc in model.NoiseComponent_list])
        else "WLS"
    )

    res = Residuals(toas, model)
    rms_res = res.calc_time_resids().to("us").value.std()
    chi2 = res.chi2
    chi2_red = res.chi2_reduced

    exclude_params = [
        "START",
        "FINISH",
        "NTOA",
        "CHI2",
        "DMDATA",
        "PSR",
        "EPHEM",
        "CLOCK",
        "UNITS",
        "TIMEEPH",
        "T2CMETHOD",
        "DILATEFREQ",
        "INFO",
        "ECL",
    ]

    exclude_prefixes = []
    if not include_dmx:
        exclude_prefixes.append("DMX")
    if not include_jumps:
        exclude_prefixes.append("JUMP")

    with StringIO("w") as tex:
        tex.write("\\documentclass{article}\n")
        tex.write("\\begin{document}\n")

        tex.write("\\begin{table}\n")
        tex.write("\\caption{Parameters for PSR %s}\n" % psrname)
        tex.write("\\begin{tabular}{ll}\n")
        tex.write("\\hline\\hline\n")
        tex.write("\\multicolumn{2}{c}{Dataset and Fit summary}\\\\ \n")
        tex.write("\\hline\n")
        tex.write("Pulsar name                  \\dotfill & %s      \\\\ \n" % psrname)
        tex.write(
            "MJD range                    \\dotfill & %d---%d \\\\ \n"
            % (mjd_start, mjd_end)
        )
        tex.write(
            "Data span (yr)               \\dotfill & %.2f    \\\\ \n" % data_span_yr
        )
        tex.write("Number of TOAs               \\dotfill & %d      \\\\ \n" % ntoas)
        tex.write("Number of free parameters    \\dotfill & %d      \\\\ \n" % nfree)
        tex.write("TOA paradigm                 \\dotfill & %s      \\\\ \n" % toa_type)
        tex.write(
            "Fitting method               \\dotfill & %s      \\\\ \n" % fit_method
        )
        tex.write("Solar system ephemeris       \\dotfill & %s      \\\\ \n" % ephem)
        tex.write("Timescale                    \\dotfill & %s      \\\\ \n" % clock)
        tex.write("Time unit                    \\dotfill & %s      \\\\ \n" % units)
        tex.write("Time ephemeris               \\dotfill & %s      \\\\ \n" % timeeph)
        tex.write("RMS TOA residuals ($\\mu s$) \\dotfill & %.2f    \\\\ \n" % rms_res)
        tex.write("chi2                         \\dotfill & %.2f    \\\\ \n" % chi2)
        tex.write("Reduced chi2                 \\dotfill & %.2f    \\\\ \n" % chi2_red)
        tex.write("\\hline\n")

        tex.write("\multicolumn{2}{c}{Measured Quantities} \\\\ \n")
        for fp in model.free_params:
            if fp not in exclude_params and all(
                [not fp.startswith(pre) for pre in exclude_prefixes]
            ):
                param = getattr(model, fp)

                if isinstance(param._parent, NoiseComponent) and not include_noise:
                    continue

                if param.value == 0 and not include_zeros:
                    continue

                if isinstance(param, MJDParameter):
                    uf = ufloat(param.value, param.uncertainty_value)
                    tex.write(
                        "%s, %s (%s)\dotfill &  %s \\\\ \n"
                        % (
                            param.name,
                            param.description,
                            str(param.units),
                            f"{uf:.1uS}",
                        )
                    )
                elif isinstance(param, maskParameter):
                    tex.write(
                        "%s %s %s, %s (%s)\dotfill &  %s \\\\ \n"
                        % (
                            param.prefix,
                            param.key,
                            " ".join(param.key_value),
                            param.description,
                            str(param.units),
                            f"{param.as_ufloat():.1uS}",
                        )
                    )
                else:
                    tex.write(
                        "%s, %s (%s)\dotfill &  %s \\\\ \n"
                        % (
                            param.name,
                            param.description,
                            str(param.units),
                            f"{param.as_ufloat():.1uS}",
                        )
                    )
        tex.write("\\hline\n")

        tex.write("\multicolumn{2}{c}{Set Quantities} \\\\ \n")
        tex.write("\\hline\n")
        for p in model.params:
            param = getattr(model, p)

            if isinstance(param._parent, NoiseComponent) and not include_noise:
                continue

            if param.value == 0 and not include_zeros:
                continue

            if (
                param.value is not None
                and param.frozen
                and p not in exclude_params
                and all([not p.startswith(pre) for pre in exclude_prefixes])
            ):
                if isinstance(param, maskParameter):
                    tex.write(
                        "%s %s %s, %s (%s)\dotfill &  %f \\\\ \n"
                        % (
                            param.prefix,
                            param.key,
                            " ".join(param.key_value),
                            param.description,
                            str(param.units),
                            param.value,
                        )
                    )
                elif isinstance(
                    param,
                    (floatParameter, AngleParameter, MJDParameter, prefixParameter),
                ):
                    tex.write(
                        "%s, %s (%s)\dotfill &  %f \\\\ \n"
                        % (param.name, param.description, str(param.units), param.value)
                    )
                elif isinstance(param, MJDParameter):
                    tex.write(
                        "%s, %s (%s)\dotfill &  %f \\\\ \n"
                        % (param.name, param.description, str(param.units), param.value)
                    )
                elif isinstance(param, strParameter):
                    tex.write(
                        "%s, %s \dotfill &  %s \\\\ \n"
                        % (param.name, param.description, param.value)
                    )
                elif isinstance(param, boolParameter):
                    tex.write(
                        "%s, %s (Y/N)\dotfill &  %s \\\\ \n"
                        % (param.name, param.description, "Y" if param.value else "N")
                    )
                elif isinstance(param, intParameter):
                    tex.write(
                        "%s, %s \dotfill &  %d \\\\ \n"
                        % (param.name, param.description, param.value)
                    )

        # Epoch of frequency determination (MJD)\dotfill & 53750 \\
        # Epoch of position determination (MJD)\dotfill & 53750 \\
        # Epoch of dispersion measure determination (MJD)\dotfill & 53750 \\
        # NE_SW (cm^-3)\dotfill & 0 \\
        # \hline
        # \multicolumn{2}{c}{Derived Quantities} \\
        # \hline
        # $\log_{10}$(Characteristic age, yr) \dotfill & 8.92 \\
        # $\log_{10}$(Surface magnetic field strength, G) \dotfill & 9.36 \\
        # $\log_{10}$(Edot, ergs/s) \dotfill & 33.46 \\
        # \hline
        # \multicolumn{2}{c}{Assumptions} \\
        # \hline
        # Clock correction procedure\dotfill & TT(BIPM2019) \\
        # Solar system ephemeris model\dotfill & DE421 \\
        # Binary model\dotfill & NONE \\
        # TDB units (tempo1 mode)\dotfill & Y \\
        # FB90 time ephemeris (tempo1 mode)\dotfill & Y \\
        # Shapiro delay due to planets\dotfill & N \\
        # Tropospheric delay\dotfill & N \\
        # Dilate frequency\dotfill & N \\
        # Electron density at 1 AU (cm$^{-3}$)\dotfill & 0.00 \\
        # Model version number\dotfill & 2.00 \\
        tex.write("\\hline\n")
        tex.write("\\end{tabular}\n")
        tex.write("\\end{table}\n")
        tex.write("\\end{document}\n")

        output = tex.getvalue()

    return output
