import spice
import numpy as np


def test_lmt(et, step):
    """
    Testing how accurate that spice can distinguish two near by times(et) with little time step
    et is the initial time
    step is the small time step
    """
    et0 = et
    et1 = et + step
    print(et0, et1, et1 - et0)
    state0, lt0 = spice.spkezr("EARTH", et0, "J2000", "NONE", "SSB")
    state1, lt1 = spice.spkezr("EARTH", et1, "J2000", "NONE", "SSB")
    diff = np.array(state0) - np.array(state1)
    print(state0, state1)
    return diff


def spice_Intplt(et, stepBitNum):
    """
    Testing interpolating spice results with two exact numbers
    """
    step = 1.0 / 2.0**stepBitNum
    numStep = 2.0**stepBitNum
    et0 = np.floor(et)  # The first integer before targeting time et
    et1 = np.ceil(et)  # The first integer after targeting time et
    exctNumArray = np.linspace(et0, et1, numStep)
    exctNumNear = min(
        exctNumArray, key=lambda x: abs(x - et)
    )  # find the closest exact number
    # Find two exact number around input time et
    if exctNumNear > et:
        exctNum0 = exctNumNear - step
        exctNum1 = exctNumNear
    elif exctNumNear == et:
        exctNum0 = et
        exctNum1 = et
    else:
        exctNum0 = exctNumNear
        exctNum1 = exctNumNear - step
    print(exctNum0, exctNum1)
    state0, lt0 = spice.spkezr("EARTH", exctNum0, "J2000", "NONE", "SSB")
    state1, lt1 = spice.spkezr("EARTH", exctNum1, "J2000", "NONE", "SSB")
    state = [
        np.interp(et, [exctNum0, exctNum1], [state0[i], state1[i]]) for i in range(6)
    ]
    lt = [np.interp(et, [exctNum0, exctNum1], [lt0, lt1])]
    stateOr, ltOr = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")

    return state, stateOr, np.array(state) - np.array(stateOr)


def spkInterp(et, stepBitNum):
    """
    This function interpolates earth state in one second with several exact points.
    To increase accuracy, each know point will be the exact number that can be represented
    by double precision.
    et is the target time
    stepBitNum determines that how small the step for exact points will be. And it is
    calculated by bits
    step  = 1.0/2.0**stepBitNum
    """
    step = 1.0 / 2.0**stepBitNum  # Step from exact point
    numStep = 2.0**stepBitNum  # Number of exact point
    et0 = np.floor(et)  # Start point of interval
    etExctArray = np.linspace(et0, et0 + 1, numStep + 1)  # Exact time point array
    stateArray = []  # Earth state array for each exact point
    ltArray = []  # light time state array for each exact point
    """
	Calculate the earth state and lt for each exact point
	"""
    for data in etExctArray:
        stateExct, ltExct = spice.spkezr("EARTH", data, "J2000", "NONE", "SSB")
        stateArray.append(stateExct)
        ltArray.append(ltExct)

    stateArray = np.array(stateArray)  # Transfer to numpy array
    ltArray = np.array(ltArray)  # Transfer to numpy array
    state = []  # Earth state for target time
    lt = []  # lt for target time
    """
	Interpolate for target time
	"""
    for i in range(6):
        state.append(np.interp(et, etExctArray, stateArray[:, i]))
        lt.append(np.interp(et, etExctArray, ltArray))
    """
	Return earth state and light time as list
	"""
    return state, lt
