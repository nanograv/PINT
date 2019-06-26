import numpy as np
import matplotlib.pyplot as plt

#assume given mean vector, covariance matrix, and fac
print("mean vector", mean_vector)
print("errors", np.sqrt(np.diag(ucov_mat)))
#scale by fac for calculation
mean_vector *= fac
ucov_mat = ((ucov_mat*fac).T*fac).T
nums = [[],[],[],[],[],[]]
for i in range(20000):
        a,b,c,d,e,f = np.random.multivariate_normal(mean_vector,ucov_mat)
        nums[0].append(a)
        nums[1].append(b)
        nums[2].append(c)
        nums[3].append(d)
        nums[4].append(e)
        nums[5].append(f)
#scale back to real units
mean_vector /= fac
for i in range(6):
    nums[i] /= fac[i]
ucov_mat = ((ucov_mat/fac).T/fac)
for i in range(6):
    data = nums[i]
    mean = np.mean(data)
    std = np.std(data)
    plt.hist(nums[i], bins=400)
    plt.title(params.keys()[i]+" mean: "+str(mean)+" std: "+str(std))
    plt.show()
    
