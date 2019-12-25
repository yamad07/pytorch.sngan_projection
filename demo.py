import matplotlib.pyplot as plt
import glob
from PIL import Image

files = glob.glob('results/*.jpg')

for file in files:
    fake = Image.open(file)
    plt.imshow(fake)
    plt.pause(0.000000001)
