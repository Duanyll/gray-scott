import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import re

def load_image(folder):
    results = []
    for file in folder.iterdir():
        if file.is_file():
            basename = file.name
            # Match this
            # f"{preset_name}_t{int(self.t)}_s{self.xgrid}_dt{self.dt:.3f}.png"
            regex = r"(.+)_t(\d+)_s(\d+)_dt(\d+\.\d+).png"
            match = re.match(regex, basename)
            if match:
                preset_name, t, s, dt = match.groups()
                t, s, dt = int(t), int(s), float(dt)
                image = Image.open(file)
                results.append((preset_name, t, s, dt, image))
            else:
                print(f"Skipping {basename}")
    return results

fig, axs = plt.subplots(3, 5, figsize=(16, 10))
folder = Path("fig5")
images = load_image(folder)
images = sorted(images, key=lambda x: x[0])
for i in range(15):
        ax = axs.flat[i]
        preset_name, t, s, dt, image = images[i]
        ax.imshow(image)
        # ax.set_title(f"{preset_name} t={t} s={s} dt={dt:.3f}")
        ax.set_title(preset_name)
        ax.axis("off")
plt.tight_layout()
plt.savefig("Figure_5.png")