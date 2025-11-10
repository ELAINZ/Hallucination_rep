from PIL import Image, ImageDraw
import numpy as np
import random, math, os, PIL

image_size = (21, 63)
save_dir = "../simple-datasets/square_bottom_left_top_center_fixed"
os.makedirs(save_dir, exist_ok=True)

num_images = 5000
area = 60
meta_data = {}
count = 0

for idx in range(num_images):
    image = Image.new('L', image_size, color=0)
    draw = ImageDraw.Draw(image)
    side_length = int(math.sqrt(area))

    if idx < num_images // 2:
        # Region 1：底部、偏左
        x_center = image_size[0] // 2
        y1 = random.randint(40, 47)
        region = "bottom_left"
    else:
        # Region 3：上方、居中
        x_center = image_size[0] // 2
        y1 = random.randint(8, 15)
        region = "top_center"

    x1 = x_center - side_length // 2
    x2 = x1 + side_length
    y2 = y1 + side_length
    draw.rectangle([x1, y1, x2, y2], fill=255, outline=255)

    img_np = np.array(image)

    new_image_array = np.zeros((64, 64), dtype=np.uint8)

    # 控制整体水平偏移 —— 把整个21宽的图放到64宽的大图偏左区域
    if region == "bottom_left":
        x_offset = 11   # 往左放一些
    else:
        x_offset = 22   # 上方几乎居中
    new_image_array[0:63, x_offset:x_offset + 21] = img_np

    final_img = PIL.Image.fromarray(new_image_array)
    filename = f"{count:05d}_{region}.png"
    final_img.save(os.path.join(save_dir, filename))

    meta_data[count] = [["square", x1, y1, side_length, region, x_offset]]
    count += 1

np.savez(os.path.join(save_dir, "meta_data.npz"), meta_data)
print(f"Generated {count} images (bottom-left & top-center, visually balanced) in {save_dir}")
