from pint.models import (
    TimingModel,
    DispersionDMX,
    FD,
    Glitch,
    PhaseJump,
    SolarWindDispersionX,
)
from pint.models.absolute_phase import AbsPhase
from pint.models.noise_model import NoiseComponent
from pint.models.parameter import (
    Parameter,
    boolParameter,
    intParameter,
    maskParameter,
    strParameter,
)
from pint.toa import TOAs
from pint.residuals import Residuals
from io import StringIO
import numpy as np


def publish_param_value(param: Parameter):
    if isinstance(param, boolParameter):
        return "Y" if param.value else "N"
    elif isinstance(param, strParameter):
        return param.value
    elif isinstance(param, intParameter):
        return str(param.value)
    else:
        return f"{param.as_ufloat():.1uS}" if not param.frozen else f"{param.value:f}"


def publish_param_unit(param: Parameter):
    return "" if param.units == "" or param.units is None else f" ({param.units})"


def publish_param(param):
    if isinstance(param, maskParameter):
        return f"{param.prefix} {param.key} {' '.join(param.key_value)}, {param.description}{publish_param_unit(param)}\dotfill &  {publish_param_value(param)} \\\\ \n"
    else:
        return f"{param.name}, {param.description}{publish_param_unit(param)}\dotfill &  {publish_param_value(param)} \\\\ \n"


def publish(
    model: TimingModel,
    toas: TOAs,
    include_dmx=False,
    include_noise=False,
    include_jumps=False,
    include_zeros=False,
    include_fd=False,
    include_glitches=False,
    include_swx=False,
    include_tzr=False,
):
    mjds = toas.get_mjds()
    mjd_start, mjd_end = int(min(mjds.value)), int(max(mjds.value))
    data_span_yr = (mjd_end - mjd_start) / 365.25

    fit_method = (
        "GLS"
        if np.any([nc.introduces_correlated_errors for nc in model.NoiseComponent_list])
        else "WLS"
    )

    res = Residuals(toas, model)

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
        "BINARY",
    ]

    exclude_components = []
    if not include_dmx:
        exclude_components.append(DispersionDMX)
    if not include_jumps:
        exclude_components.append(PhaseJump)
    if not include_fd:
        exclude_components.append(FD)
    if not include_noise:
        exclude_components.append(NoiseComponent)
    if not include_glitches:
        exclude_components.append(Glitch)
    if not include_swx:
        exclude_components.append(SolarWindDispersionX)
    if not include_tzr:
        exclude_components.append(AbsPhase)

    with StringIO("w") as tex:
        tex.write("\\documentclass{article}\n")
        tex.write("\\begin{document}\n")

        tex.write("\\begin{table}\n")
        tex.write("\\caption{Parameters for PSR %s}\n" % model.PSR.value)
        tex.write("\\begin{tabular}{ll}\n")
        tex.write("\\hline\\hline\n")
        tex.write("\\multicolumn{2}{c}{Dataset and model summary}\\\\ \n")
        tex.write("\\hline\n")
        tex.write(
            f"Pulsar name                  \\dotfill & {model.PSR.value}      \\\\ \n"
        )
        tex.write(
            f"MJD range                    \\dotfill & {mjd_start}---{mjd_end} \\\\ \n"
        )
        tex.write(
            f"Data span (yr)               \\dotfill & {data_span_yr:.2f}    \\\\ \n"
        )
        tex.write(f"Number of TOAs               \\dotfill & {len(toas)}      \\\\ \n")
        tex.write(
            f"TOA paradigm                 \\dotfill & {'Wideband' if toas.is_wideband() else 'Narrowband'}      \\\\ \n"
        )
        tex.write(
            f"Solar system ephemeris       \\dotfill & {model.EPHEM.value}      \\\\ \n"
        )
        tex.write(
            f"Timescale                    \\dotfill & {model.CLOCK.value}      \\\\ \n"
        )
        tex.write(
            f"Time unit                    \\dotfill & {model.UNITS.value}      \\\\ \n"
        )
        tex.write(
            f"Time ephemeris               \\dotfill & {model.TIMEEPH.value}      \\\\ \n"
        )

        if model.BINARY.value is not None:
            tex.write(
                f"Binary model               \\dotfill & {model.BINARY.value}      \\\\ \n"
            )

        if "PhaseJump" in model.components:
            tex.write(
                f"Number of JUMPs               \\dotfill & {model.get_number_of_jumps()}      \\\\ \n"
            )

        if "DispersionDMX" in model.components:
            tex.write(
                f"Number of DMX ranges          \\dotfill & {len(model.components['DispersionDMX'].get_indices())}      \\\\ \n"
            )

        if "SolarWindDispersionX" in model.components:
            tex.write(
                f"Number of SWX ranges          \\dotfill & {len(model.components['SolarWindDispersionX'].get_indices())}      \\\\ \n"
            )

        if "Glitch" in model.components:
            tex.write(
                f"Number of Glitches          \\dotfill & {len(model.components['Glitch'].glitch_indices)}      \\\\ \n"
            )

        if "FD" in model.components:
            tex.write(
                f"Number of FD parameters          \\dotfill & {model.num_FD_terms}      \\\\ \n"
            )

        if "ScaleToaError" in model.components:
            tex.write(
                f"Number of EFACs          \\dotfill & {len(model.EFACs)}     \\\\ \n"
            )
            tex.write(
                f"Number of EQUADs          \\dotfill & {len(model.EQUADs)}      \\\\ \n"
            )

        if "EcorrNoise" in model.components:
            tex.write(
                f"Number of ECORRs          \\dotfill & {len(model.ECORRs)}      \\\\ \n"
            )

        if "ScaleDmError" in model.components:
            tex.write(
                f"Number of DMEFACs          \\dotfill & {len(model.DMEFACs)}      \\\\ \n"
            )
            tex.write(
                f"Number of DMEQUADs          \\dotfill & {len(model.DMEQUADs)}      \\\\ \n"
            )

        tex.write("\\hline\n")

        tex.write("\\multicolumn{2}{c}{Fit summary}\\\\ \n")
        tex.write("\\hline\n")
        tex.write(
            f"Number of free parameters    \\dotfill & {len(model.free_params)}      \\\\ \n"
        )
        tex.write(f"Fitting method               \\dotfill & {fit_method}      \\\\ \n")
        tex.write(
            f"RMS TOA residuals ($\\mu s$) \\dotfill & {res.calc_time_resids().to('us').value.std():.2f}   \\\\ \n"
        )
        tex.write(f"chi2                         \\dotfill & {res.chi2:.2f}    \\\\ \n")
        tex.write(
            f"Reduced chi2                 \\dotfill & {res.chi2_reduced:.2f}    \\\\ \n"
        )
        tex.write("\\hline\n")

        tex.write("\multicolumn{2}{c}{Measured Quantities} \\\\ \n")
        for fp in model.free_params:
            param = getattr(model, fp)
            if (
                fp not in exclude_params
                and all(
                    [not isinstance(param._parent, exc) for exc in exclude_components]
                )
                and (param.value != 0 or include_zeros)
            ):
                tex.write(publish_param(param))

        tex.write("\\hline\n")

        tex.write("\multicolumn{2}{c}{Set Quantities} \\\\ \n")
        tex.write("\\hline\n")
        for p in model.params:
            param = getattr(model, p)

            if (
                param.value is not None
                and param.frozen
                and p not in exclude_params
                and all(
                    [not isinstance(param._parent, exc) for exc in exclude_components]
                )
                and (param.value != 0 or include_zeros)
            ):
                tex.write(publish_param(param))

        tex.write("\\hline\n")
        tex.write("\\end{tabular}\n")
        tex.write("\\end{table}\n")
        tex.write("\\end{document}\n")

        output = tex.getvalue()

    return output
