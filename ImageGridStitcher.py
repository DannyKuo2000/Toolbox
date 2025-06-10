from PIL import Image
import numpy as np
import os

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

        # ✅ NEW: crop to fit box (to prevent overlap)
        if resized.width > box_width or resized.height > box_height:
            left = (resized.width - box_width) // 2
            top = (resized.height - box_height) // 2
            right = left + box_width
            bottom = top + box_height
            resized = resized.crop((left, top, right, bottom))

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
    '''layout = {
    os.path.join(input_dir, "Dusa.png"): {"pos": (0, 0), "span": (1, 1), "scale": 1.0},  
    os.path.join(input_dir, "Berserker_cute.png"): {"pos": (0, 1), "span": (1, 1), "scale": 1.0},
    os.path.join(input_dir, "Reaper_cute.png"): {"pos": (0, 2), "span": (1, 2), "scale": 1.0},
    }
    grid_size = (1, 4)
    cell_size = (1920*2//4, 1080)'''
    layout = {
    os.path.join(input_dir, "IMG_0524.png"): {"pos": (0, 0), "span": (1, 1), "scale": 1.0},  
    os.path.join(input_dir, "IMG_0525.png"): {"pos": (0, 1), "span": (1, 1), "scale": 1.0},
    os.path.join(input_dir, "IMG_0528.png"): {"pos": (0, 2), "span": (1, 1), "scale": 1.0},
    os.path.join(input_dir, "IMG_0529.png"): {"pos": (0, 3), "span": (1, 1), "scale": 1.0},
    os.path.join(input_dir, "IMG_0516.png"): {"pos": (0, 4), "span": (1, 1), "scale": 1.0},  
    os.path.join(input_dir, "IMG_0517.png"): {"pos": (0, 5), "span": (1, 1), "scale": 1.0},
    os.path.join(input_dir, "IMG_0520.png"): {"pos": (0, 6), "span": (1, 1), "scale": 1.0},
    os.path.join(input_dir, "IMG_0531.png"): {"pos": (0, 7), "span": (1, 1), "scale": 1.0},
    }
    grid_size = (1, 8)
    cell_size = (1920*2//8, 1080)
    final = create_layout(layout, grid_size, cell_size, resize_mode='fill')
    final.save(os.path.join(output_dir, "output_wallpaper_7.png"))
    print(f"✅ 拼接完成")
