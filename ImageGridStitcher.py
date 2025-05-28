from PIL import Image
import numpy as np
import os

def resize_and_place(image, box_size, mode='fill'):
    # Resize image to fit in box_size (width, height)
    img_w, img_h = image.size
    box_w, box_h = box_size

    if mode == 'fit':
        # 等比例縮放，圖片完整顯示在區塊內，會留空白
        scale = min(box_w / img_w, box_h / img_h)
        new_size = (int(img_w * scale), int(img_h * scale))
        image_resized = image.resize(new_size, Image.LANCZOS)
        background = Image.new('RGB', box_size, (0, 0, 0))
        offset = ((box_w - new_size[0]) // 2, (box_h - new_size[1]) // 2)
        background.paste(image_resized, offset)
        return background

    elif mode == 'fill':
        # 等比例縮放，圖片填滿整個區塊，會裁切
        scale = max(box_w / img_w, box_h / img_h)
        new_size = (int(img_w * scale), int(img_h * scale))
        image_resized = image.resize(new_size, Image.LANCZOS)
        left = (new_size[0] - box_w) // 2
        top = (new_size[1] - box_h) // 2
        image_cropped = image_resized.crop((left, top, left + box_w, top + box_h))
        return image_cropped

    elif mode == 'stretch':
        # 非等比例拉伸到剛好填滿
        return image.resize(box_size, Image.LANCZOS)

    else:
        raise ValueError("Unsupported resize mode. Choose from 'fit', 'fill', 'stretch'.")

def create_layout(layout, grid_size, cell_size, resize_mode='fit', background_color=(0, 0, 0)):
    from PIL import Image
    import os

    rows, cols = grid_size
    canvas = Image.new("RGB", (cols * cell_size[0], rows * cell_size[1]), background_color)

    for image_path, config in layout.items():
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Cannot open {image_path}: {e}")
            continue

        pos = config["pos"]
        span = config.get("span", (1, 1))
        scale = config.get("scale", 1.0)

        x0 = pos[1] * cell_size[0]
        y0 = pos[0] * cell_size[1]
        box_width = span[1] * cell_size[0]
        box_height = span[0] * cell_size[1]

        center_x = x0 + box_width // 2
        center_y = y0 + box_height // 2

        if resize_mode == 'fit':
            img.thumbnail((box_width, box_height), Image.LANCZOS)
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            resized = img.resize((new_width, new_height), Image.LANCZOS)
        elif resize_mode == 'fill':
            scale_fill = max(box_width / img.width, box_height / img.height)
            resized = img.resize((int(img.width * scale_fill * scale), int(img.height * scale_fill * scale)), Image.LANCZOS)
        elif resize_mode == 'stretch':
            resized = img.resize((int(box_width * scale), int(box_height * scale)), Image.LANCZOS)
        else:
            raise ValueError(f"Unsupported resize mode: {resize_mode}")

        paste_x = center_x - resized.width // 2
        paste_y = center_y - resized.height // 2
        canvas.paste(resized, (paste_x, paste_y))

    return canvas


# Example usage
if __name__ == '__main__':
    input_dir = "C:\\Users\\Danny\\Desktop\\Wallpaper"
    output_dir = "C:\\Users\\Danny\\Desktop\\Wallpaper"
    """layout = {  #  
    os.path.join(input_dir, "Ranger.png"): {"pos": (0, 0), "span": (2, 1), "scale": 1.0},
    os.path.join(input_dir, "Reaper.png"): {"pos": (0, 1), "span": (2, 1), "scale": 1.0},
    os.path.join(input_dir, "Pyromancer.png"): {"pos": (0, 2), "span": (2, 1), "scale": 1.0},
    os.path.join(input_dir, "Berserker.png"): {"pos": (0, 3), "span": (2, 1), "scale": 1.0},  # 沒給 scale 就用 1.0
    os.path.join(input_dir, "Reaper_cute.png"): {"pos": (0, 4), "span": (2, 2), "scale": 1.0},
    }
    grid_size = (2, 6)
    cell_size = (1920*2//6, 1080//2)  # or 1920x1080 etc."""
    layout = {
    os.path.join(input_dir, "Dusa.png"): {"pos": (0, 0), "span": (1, 1), "scale": 1.0},  
    os.path.join(input_dir, "Berserker_cute.png"): {"pos": (0, 1), "span": (1, 1), "scale": 1.0},
    os.path.join(input_dir, "Reaper_cute.png"): {"pos": (0, 2), "span": (1, 2), "scale": 1.0},
    }
    grid_size = (1, 4)
    cell_size = (1920*2//4, 1080)
    """layout = {
    os.path.join(input_dir, "Ranger.png"): {"pos": (0, 0), "span": (1, 2), "scale": 1.0},
    os.path.join(input_dir, "Berserker_cute.png"): {"pos": (0, 1), "span": (2, 2), "scale": 1.5},  # 沒給 scale 就用 1.0
    os.path.join(input_dir, "Reaper.png"): {"pos": (0, 3), "span": (1, 2), "scale": 1.0},
    os.path.join(input_dir, "Pyromancer.png"): {"pos": (0, 4), "span": (1, 2), "scale": 1.0},
    os.path.join(input_dir, "Reaper_cute.png"): {"pos": (0, 5), "span": (2, 2), "scale": 1.5},
    os.path.join(input_dir, "Berserker.png"): {"pos": (0, 7), "span": (1, 1), "scale": 1.0},
    }
    grid_size = (2, 8)
    cell_size = (1920*2//8, 1080//2)"""
    final = create_layout(layout, grid_size, cell_size, resize_mode='fill')
    final.save(os.path.join(output_dir, "output_wallpaper_2.png"))
    print(f"✅ 拼接完成")
