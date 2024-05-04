from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')
image_path = "dataset/train/bees_image/36900412_92b81831ad.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("test", img_array, 3, dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()
