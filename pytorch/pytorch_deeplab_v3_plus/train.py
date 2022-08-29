from PIL import Image
import numpy as np
import  pycave_pkg.pycave.bayes.gmm as gmm

img = Image.open("./2007_000032.jpg", mode="RGB")
nd_img = np.array(img)
model = gmm.GaussianMixture(num_components=10, covariance_type="full", init_strategy="random", covariance_regularization=25)
model.fit(nd_img)
print(model.score_samples(np.array([0,0,0])))

