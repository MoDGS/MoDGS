import matplotlib.pyplot as plt
import numpy as np

# Define the size of the image
image_size = 240

# Create an empty image
image = np.zeros((image_size, image_size, 3))

# Define the colors for the squares
colors = [
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [1, 1, 0],  # Yellow
    [1, 0, 1],  # Magenta
    [0, 1, 1]   # Cyan
]

# Define the size of the squares
square_size = image_size // len(colors)

# Fill the image with colored squares
for i, color in enumerate(colors):
    start = i * square_size
    end = start + square_size
    image[start:end, start:end] = color
import PIL
PIL.Image.fromarray((image * 255).astype(np.uint8)).save('test.png')
# Display the image
flow=np.ones((image_size, image_size, 2))*40
flow[:, :, 0] = 0
import cv2
def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]
    print(flow_new[0,0])
    # res = cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    res = cv2.remap(img, flow_new[...,0], flow_new[...,1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return res
warped_img = warp_flow(image, flow.astype(np.float32))
PIL.Image.fromarray((warped_img * 255).astype(np.uint8)).save('test_warped.png')

plt.imshow(image)

plt.show()