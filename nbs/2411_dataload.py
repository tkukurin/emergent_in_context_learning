# uv pip install ipykernel matplotlib
# %%
import tk
import importlib
importlib.reload(tk)
print(tk.envsetup())
# %%
from eicl.datasets import data_generators as dglib
dsomni = dglib.OmniglotDatasetForSampling('train')
# %%
import matplotlib.pyplot as plt
plt.imshow(dsomni.data[0])
# %%
pls, axs = plt.subplots(2, 2)
for i, a in enumerate(axs.flatten()):
    a.imshow(dsomni.data[i])
# %%