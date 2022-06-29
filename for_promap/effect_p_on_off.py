import numpy as np
from TraceModel import TraceModel
from FluorescenceModel import ModelParams
import seaborn as sns

params = ModelParams(0.1, 0.1, 1, 0.1, 0.2, 1)

trace = TraceModel(params, 0.1, 1000)

true_trace = trace.generate_trace(1)
probs = np.zeros((20,20))
p_ons = np.linspace(0.00001, 0.2, 20)
p_offs = np.linspace(0.00001, 0.2, 20)
for i in range(probs.shape[0]):
    for j in range(probs.shape[1]):
        params = ModelParams(p_ons[i], p_offs[j], 1, 0.1, 0.2, 1)
        test_trace = TraceModel(params, 0.1, 1000)
        prob = test_trace.p_trace_given_y(true_trace, 1)
        print(prob)
        probs[i,j] = prob

        
ax = sns.heatmap(probs, vmax = -1704, vmin = -1720, xticklabels=np.around(p_ons,2),
                 yticklabels=np.around(p_offs,2), cmap='icefire', linewidth=1)


a = probs
a[a> -1705] = 5
a[a>-1715] = 4
a[a>-1720] = 3
a[a>-1730] = 2
a[a>-1750] = 1