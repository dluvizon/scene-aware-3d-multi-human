import numpy as np
from pathlib import Path
from PIL import Image
import torch

def io_mkdir(newpath):
    try:
        path = Path(newpath)
        path.mkdir(parents=True, exist_ok=True)
    except:
        print (f'Error trying to create path {newpath}')
        raise

def save_image(img, filename):
    if torch.is_tensor(img):
        if img.is_cuda:
            img = img.cpu().detach().numpy()
        else:
            img = img.detach().numpy()
    if isinstance(img, np.ndarray):
        img = (255.9 * img).astype(np.uint8)
        im = Image.fromarray(img)
    else:
        raise ValueError(f'Invalid input type {type(img)}!')

    im.save(filename)
