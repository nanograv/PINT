# Test timing model functions to test out the residuals class

def F0(toa,model):

    dt = (toa - model.PEPOCH.value)*24*3600
    ph = dt*model.F0.value

    return ph


