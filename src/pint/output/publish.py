"""Generate LaTeX summary of a timing model and TOAs."""
from pint.models import (
    TimingModel,
    DispersionDMX,
    FD,
    Glitch,
    PhaseJump,
    SolarWindDispersionX,
    AbsPhase,
    Wave,
)
from pint.models.dispersion_model import DispersionJump
from pint.models.noise_model import NoiseComponent
from pint.models.parameter import (
    Parameter,
    funcParameter,
)
from pint.toa import TOAs
from pint.residuals import Residuals, WidebandTOAResiduals
from io import StringIO
import numpy as np


def publish_param(param: Parameter):
    """Return LaTeX line for a parameter"""
    label, value = param.as_latex()
    return f"{label}\\dotfill &  {value} \\\\ \n"


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
    include_prefix_summary=True,
    include_set_params=True,
    include_derived_params=True,
    include_fit_summary=True,
):
    """Generate LaTeX summary of a given timing model and TOAs.

    Parameters
    ----------
    model: pint.model.timing_model.TimingModel
        Input timing model
    toas: TOAs
        Input TOAs
    include_dmx: bool
        Whether to include DMX paremeters (default is False)
    include_noise: bool
        Whether to include noise paremeters (default is False)
    include_jumps: bool
        Whether to include jump paremeters (JUMPs, DMJUMPs) (default is False)
    include_zeros: bool
        Whether to include paremeters which are zero (default is False)
    include_fd: bool
        Whether to include FD paremeters (default is False)
    include_glitches: bool
        Whether to include glitch paremeters (default is False)
    include_swx: bool
        Whether to include SWX paremeters (default is False)
    include_tzr: bool
        Whether to include TZR paremeters (default is False)
    include_prefix_summary: bool
        Whether to include a summary of prefix and mask parameters (default is True)
    include_set_params: bool
        Whether to include set params (default is True)
    include_derived_params: bool
        Whether to include derived params (default is True)
    include_fit_summary: bool
        Whether to include fit summary params (default is True)

    Returns
    -------
    latex_summary: str
        The LaTeX summary
    """
    mjds = toas.get_mjds()
    mjd_start, mjd_end = int(min(mjds.value)), int(max(mjds.value))
    data_span_yr = (mjd_end - mjd_start) / 365.25

    fit_method = (
        "GLS"
        if np.any([nc.introduces_correlated_errors for nc in model.NoiseComponent_list])
        else "WLS"
    )

    if toas.is_wideband():
        res = WidebandTOAResiduals(toas, model)
        toares = res.toa
        dmres = res.dm
    else:
        res = Residuals(toas, model)
        toares = res

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

    exclude_components = [Wave]
    if not include_dmx:
        exclude_components.append(DispersionDMX)
    if not include_jumps:
        exclude_components.extend([PhaseJump, DispersionJump])
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
            f"Pulsar name                  \\dotfill & {model.PSR.value.replace('-','$-$')}      \\\\ \n"
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

        if include_prefix_summary:
            if "PhaseJump" in model.components:
                tex.write(
                    f"Number of JUMPs               \\dotfill & {model.get_number_of_jumps()}      \\\\ \n"
                )

            if "DispersionJump" in model.components:
                tex.write(
                    f"Number of DMJUMPs               \\dotfill & {len(model.dm_jumps)}      \\\\ \n"
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
            if "Wave" in model.components:
                tex.write(
                    f"Number of WAVE components    \\dotfill & {model.num_wave_terms}      \\\\ \n"
                )

        tex.write("\\hline\n")

        if include_fit_summary:
            tex.write("\\multicolumn{2}{c}{Fit summary}\\\\ \n")
            tex.write("\\hline\n")
            tex.write(
                f"Number of free parameters    \\dotfill & {len(model.free_params)}      \\\\ \n"
            )
            tex.write(
                f"Fitting method               \\dotfill & {fit_method}      \\\\ \n"
            )
            tex.write(
                f"RMS TOA residuals ($\\mu s$) \\dotfill & {toares.calc_time_resids().to('us').value.std():.2f}   \\\\ \n"
            )
            if toas.is_wideband():
                tex.write(
                    f"RMS DM residuals (pc / cm3) \\dotfill & {dmres.calc_resids().to('pc/cm^3').value.std():.2f}   \\\\ \n"
                )
            tex.write(
                f"$\\chi^2$                         \\dotfill & {res.chi2:.2f}    \\\\ \n"
            )
            if toas.is_wideband():
                tex.write(f"Degrees of freedom \\dotfill & {res.dof}   \\\\ \n")
            else:
                tex.write(
                    f"Reduced $\\chi^2$                 \\dotfill & {res.reduced_chi2:.2f}    \\\\ \n"
                )
            tex.write("\\hline\n")

        tex.write("\multicolumn{2}{c}{Measured Quantities} \\\\ \n")
        tex.write("\\hline\n")
        for fp in model.free_params:
            param = getattr(model, fp)
            if (
                all([not isinstance(param._parent, exc) for exc in exclude_components])
                and fp not in exclude_params
                and (param.value != 0 or include_zeros)
            ):
                tex.write(publish_param(param))

        tex.write("\\hline\n")

        if include_set_params:
            tex.write("\multicolumn{2}{c}{Set Quantities} \\\\ \n")
            tex.write("\\hline\n")
            for p in model.params:
                param = getattr(model, p)

                if (
                    all(
                        [
                            not isinstance(param._parent, exc)
                            for exc in exclude_components
                        ]
                    )
                    and param.value is not None
                    and param.frozen
                    and p not in exclude_params
                    and (param.value != 0 or include_zeros)
                    and not isinstance(param, funcParameter)
                ):
                    tex.write(publish_param(param))

            tex.write("\\hline\n")

        if include_derived_params:
            derived_params = [
                getattr(model, p)
                for p in model.params
                if isinstance(getattr(model, p), funcParameter)
                and getattr(model, p).quantity is not None
            ]
            if len(derived_params) > 0:
                tex.write("\multicolumn{2}{c}{Derived Quantities} \\\\ \n")
                tex.write("\\hline\n")
                for param in derived_params:
                    tex.write(publish_param(param))
                tex.write("\\hline\n")

        tex.write("\\end{tabular}\n")
        tex.write("\\end{table}\n")
        tex.write("\\end{document}\n")

        output = tex.getvalue()

    output = output.replace("_", "\\_")

    return output
