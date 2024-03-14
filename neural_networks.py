from PIL import Image
import numpy as np
import os


dir_path = 'C:\\Users\\avina\\PycharmProjects\\ManasTaskphase\\NNtrain\\bart_simpson'
files = os.listdir(dir_path)
for file in files:
    file_path = os.path.join(dir_path, file)
    bw_image = Image.open(file_path).convert('L')
    bw_array = np.array(bw_image)
    print(bw_array)

