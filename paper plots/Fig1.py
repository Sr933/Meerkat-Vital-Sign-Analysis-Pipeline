import matplotlib.pyplot as plt
import cv2
import meerkat
import numpy as np

side_view_file=r"C:\Users\silas\Downloads\IMG20231031141142_2.jpg"

img = cv2.imread(side_view_file)

file_1=r"C:\Users\silas\Master Project\meerkat raw data\example_files\mk501\file_20231005_153246.mkt"


reader = meerkat.MktReader(
    file_1,
    auto_fix_timestamp=True,
)
colour, depth, ir, params = reader.read_frame()


# Plot data
plt.style.use(['default'])
params = {"ytick.color" : "black",
    "xtick.color" : "black",
    "axes.labelcolor" : "black",
    "axes.edgecolor" : "black",
    "text.usetex" : False,
    "font.family" : "serif",
    "font.sans-serif": "Helvetica",
    }
plt.rcParams.update(params)
fig, axs = plt.subplots(2, 2, figsize=(8.27, 4.5))


axs[0,0].imshow(img[:, :, [2, 1, 0]])
axs[0,0].set_axis_off()
axs[0,0].set_title("(a) Recording setup", fontsize=12)

ir=cv2.rectangle(ir, (750, 300), (820, 420), (0, 0, 0), -1)
axs[1,0].imshow(ir)
axs[1,0].set_axis_off()
axs[1,0].set_title("(c) Infrared image", fontsize=12)

colour=cv2.rectangle(colour, (440, 260), (475, 310), (0, 0, 0), -1)
colour=cv2.rectangle(colour, (750, 300), (820, 420), (0, 0, 0), -1)
axs[0,1].imshow(colour[:, :, [2, 1, 0]])
axs[0,1].set_axis_off()
axs[0,1].set_title("(b) RGB image", fontsize=12)

axs[1,1].imshow(depth)
axs[1,1].set_axis_off()
axs[1,1].set_title("(d) Depth image", fontsize=12)


plt.tight_layout(pad=1)
plt.show()
