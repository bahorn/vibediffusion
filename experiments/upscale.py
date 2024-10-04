import sys
from tqdm import tqdm
import onnxruntime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def to_numpy(img):
    return np.asarray(img).astype(np.float32) / 255.0


session = onnxruntime.InferenceSession('models/4xNomos2_hq_dat2_fp32.onnx')

img = Image.open(sys.argv[1]).resize((128, 128))
img = to_numpy(img)
img = img.transpose(2, 0, 1)

print('here')

for i in tqdm(range(4)):
    res = session.run([], {'input': [img]})[0][0]
import matplotlib.pyplot as plt

print(res.shape)
res = res.transpose(1, 2, 0)
print(res.shape)

plt.imshow(res.astype(np.float32))
plt.show()
