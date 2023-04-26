import pint.models.model_builder as mb
import pint.toa as toa
import pint.fitter as fitter

# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import astropy.units as u
import numpy as np
import os
from tempfile import mkstemp
from shutil import move, copyfile
from os import remove, close
import pint.utils as ut
import subprocess
import re
import tempo2_utils as t2u
from astropy.table import Table


def change_parfile(filename, param, value):
    """A function to change parfile parameter value and fit flag"""
    strv = ut.longdouble2string(value)
    fh, abs_path = mkstemp()
    with open(abs_path, "w") as new_file:
        with open(filename) as old_file:
            for line in old_file:
                if line.startswith(f"{param} "):
                    l = line.split()
                    line = line.replace(l[1], strv)
                    line = line.replace(" 0 ", " 1 ")
                new_file.write(line)
    close(fh)
    # Remove original file
    remove(filename)
    # Move new file
    move(abs_path, filename)


def reset_model(source, target, fitter):
    """Change parfile and fitter to original parameter value."""
    fitter.reset_model()
    copyfile(source, target)


def perturb_param(f, param, h, source, target):
    """Perturbate parameter value and change the corresponding par file"""
    reset_model(source, target, f)
    pn = f.model.match_param_aliases(param)
    if pn != "":
        par = getattr(f.model, pn)
        orv = par.value
        par.value = (1 + h) * orv
        print(param, " New value ", par.value)
        f.update_resids()
        f.set_fitparams(pn)
        change_parfile(target, param, par.value)
    return f


def check_tempo_output(parf, timf, result_par):
    """Check out tempo output"""
    a = subprocess.check_output(f"tempo -f {parf} {timf}", shell=True)
    tempo_m = mb.get_model(result_par)
    info_idx = a.index("Weighted RMS residual: pre-fit")
    res = a[info_idx:-1]
    mpre = re.search("pre-fit(.+?)us", res)
    mpost = re.search("post-fit(.+?)us", res)
    mchi = re.search("=(.+?)pre/post", res)
    try:
        pre_rms = float(mpre[1])
        post_rms = float(mpost[1])
        chi = float(mchi[1])
    except ValueError:
        pre_rms = mpre[1]
        post_rms = mpost[1]
        chi = mchi[1]
        if chi.startswith("**"):
            chi = 0.0
    return tempo_m, pre_rms, post_rms, chi


def check_tempo2_output(parf, timf, p, result_par):
    """Check out tempo2 output"""
    res = subprocess.check_output(
        f"tempo2 -f {parf} {timf} -norescale -newpar", shell=True
    )
    mpre = re.search("RMS pre-fit residual =(.+?)(us)", res)
    mpost = re.search("RMS post-fit residual =(.+?)(us)", res)
    mchi = re.search("Chisqr/nfree =(.+?)/", res)
    m = mb.get_model(result_par)
    mpostn = re.findall(r"\d+\.\d+", mpost[1])
    try:
        pre_rms = float(mpre[1])
    except ValueError:
        pre_rms = mpre[1]
    try:
        post_rms = float(mpostn[0])
    except ValueError:
        post_rms = mpost[1]
    try:
        chi = float(mchi[1]) / len(t.table)
    except Exception:
        chi = mchi[1]
    pv = getattr(m, p).value
    pu = getattr(m, p).uncertainty_value
    return pv, pu, post_rms, chi


if __name__ == "__main__":
    fs = os.listdir(".")
    parfiles = [x for x in fs if x.endswith(".par")]
    timfiles = [x for x in fs if x.endswith(".tim")]
    # Get the data file name base
    base = []
    for fn in parfiles:
        b = fn.replace("_ori.par", "")
        if not b.endswith(".par") and b not in base:
            base.append(b.replace(".gls", ""))

    per_step = {
        "A1": 1e-05,
        "LAMBDA": 1e-09,
        "DMX_0003": 2,
        "E": 0.2,
        "F0": 1e-12,
        "F1": 0.01,
        "M2": 10.0,
        "OM": 1e-06,
        "PB": 1e-08,
        "PMLAMBDA": 0.001,
        "PMBETA": 0.1,
        "PX": 100,
        "BETA": 1e-5,
        "SINI": -0.5075,
        "T0": 1e-10,
        "FD1": 1e-2,
        "EDOT": 1e6,
        "TASC": 1e-10,
        "EPS1": 1e-2,
        "EPS2": 1e-2,
    }
    res = {}

    for b_name in base:
        # if not b_name.startswith('B1855'):
        #     continue
        if b_name.endswith("+12"):
            par = f"{b_name}_ori.par"
            tempo_par = f"{b_name}_ptb.par"
        else:
            par = f"{b_name}.gls_ori.par"
            tempo_par = f"{b_name}.gls_ptb.par"
        tim = f"{b_name}.tim"
        m = mb.get_model(par)
        result_par = f"{m.PSR.value}.par"
        res[b_name] = {}
        cmd = subprocess.list2cmdline(["grep", "'EPHEM'", par])
        out = subprocess.check_output(cmd, shell=True)
        out = out.split()
        t = toa.get_TOAs(tim, ephem=out[1].lower())
        f = fitter.WlsFitter(t, m)
        ori_resd = f.resids.calc_time_resids(False)
        ori_rms = ori_resd.std().to("us").value
        ori_red_chi = f.resids.chi2_reduced

        pp_row = []
        pt_row = []
        pt2_row = []
        for p, v in zip(per_step.keys(), per_step.values()):
            # for p,v in zip(['E',], [per_step['E'],]):
            pn = f.model.match_param_aliases(p)
            # if pn != 'ECC':
            #      continue
            if pn != "":
                print(pn)
                p_init = getattr(f.model_init, pn)
                p_ori = p_init.value
                p_ori_unc = p_init.uncertainty_value
                if p_init.quantity is None:
                    continue
                if p_init.uncertainty_value is None:
                    continue
                delay = f.model.delay(f.toas.table)
                f = perturb_param(f, p, v, par, tempo_par)
                f.fit_toas()

                try:
                    T2_pval, T2_unc, T2_post_rms, T2_chi = check_tempo2_output(
                        tempo_par, tim, p, result_par
                    )
                except:
                    T2_pval, T2_unc, T2_post_rms, T2_chi = check_tempo2_output(
                        tempo_par, tim, pn, result_par
                    )
                # Tempo_fitting
                T_m, T_pre_rms, T_post_rms, T_chi = check_tempo_output(
                    tempo_par, tim, result_par
                )
                p_fit = getattr(f.model, pn)
                p_fit_value = p_fit.value
                p_fit_unc = p_fit.uncertainty_value
                resd = f.resids.calc_time_resids(False)
                P_rms = resd.std().to("us").value
                P_red_chi = f.resids.chi2
                p_tempo = getattr(T_m, pn).value
                p_tempo_unc = getattr(T_m, pn).uncertainty_value
                print("Tempo", getattr(T_m, pn))
                print("Tempo2", T2_unc)

                diff_PINT_T = np.abs(p_fit_value - p_tempo)
                diff_PINT_T2 = np.abs(p_fit_value - T2_pval)

                if p_tempo_unc is None:
                    rela_diff_T = 0.0
                    ucdiff_T = 0.0
                else:
                    rela_diff_T = diff_PINT_T / np.abs(p_tempo_unc)
                    ucdiff_T = p_fit_unc / p_tempo_unc
                if T2_unc is None:
                    rela_diff_T2 = 0.0
                    ucdiff_T2 = 0.0
                else:
                    rela_diff_T2 = diff_PINT_T2 / np.abs(T2_unc)
                    ucdiff_T2 = p_fit_unc / T2_unc
                pt_row.append(
                    (
                        pn,
                        p_fit_value,
                        p_tempo,
                        diff_PINT_T,
                        rela_diff_T,
                        p_fit_unc,
                        p_tempo_unc,
                        ucdiff_T,
                        P_red_chi,
                        T_chi * f.toas.ntoas,
                    )
                )

                pt2_row.append(
                    (
                        pn,
                        p_fit_value,
                        T2_pval,
                        diff_PINT_T2,
                        rela_diff_T2,
                        p_fit_unc,
                        T2_unc,
                        ucdiff_T2,
                        P_red_chi,
                        T2_chi * f.toas.ntoas,
                    )
                )
                print(
                    "Tempo row",
                    (
                        pn,
                        p_fit_value,
                        p_tempo,
                        diff_PINT_T,
                        rela_diff_T,
                        p_fit_unc,
                        p_tempo_unc,
                        ucdiff_T,
                        P_red_chi,
                        T_chi * f.toas.ntoas,
                    ),
                )
                print(
                    "Tempo2 row",
                    (
                        pn,
                        p_fit_value,
                        T2_pval,
                        diff_PINT_T2,
                        rela_diff_T2,
                        p_fit_unc,
                        T2_unc,
                        ucdiff_T2,
                        P_red_chi,
                        T2_chi * f.toas.ntoas,
                    ),
                )

        if pt_row != []:
            pt_table = Table(
                rows=pt_row,
                names=(
                    "Parameter",
                    "PINT_postFit_value",
                    "T_value",
                    "par_diff(PINT-T)",
                    "relative_diff_(PINT-T)/Tempo_uncertainty",
                    "PINT_postFit_uncertainty",
                    "Tempo_uncertainty",
                    "PINT_uncertainty/tempo_uncertainty",
                    "PINT_chi",
                    "T_chi",
                ),
                meta={"name": "PINT_Tempo result table"},
                dtype=("S10", "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8"),
            )
        if pt2_row != []:
            pt2_table = Table(
                rows=pt2_row,
                names=(
                    "Parameter",
                    "PINT_postFit_value",
                    "T2_value",
                    "par_diff(PINT-T2)",
                    "relative_diff_(PINT-T2)/Tempo2_uncertainty",
                    "PINT_postFit_uncertainty",
                    "Tempo2_uncertainty",
                    "PINT_uncertainty/tempo2_uncertainty",
                    "PINT_chi",
                    "T2_chi",
                ),
                meta={"name": "PINT_Tempo2 result table"},
                dtype=("S10", "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8"),
            )

        # res[b_name] = (pt_table, pt2_table)
        out_name_pt = f"{b_name}_PT.html"
        print(f"Write {out_name_pt}")
        pt_table.write(out_name_pt, format="ascii.html", overwrite=True)
        out_name_pt2 = f"{b_name}_PT2.html"
        print(f"Write {out_name_pt2}")
        pt2_table.write(out_name_pt2, format="ascii.html", overwrite=True)
        reset_model(par, tempo_par, f)
    # subprocess.call("cat " + out_name_pt + ">>" + out_name_pt, shell=True)
