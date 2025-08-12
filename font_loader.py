import os
import matplotlib.font_manager as fm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import ImageFont

def load_fonts(font_dir=None):
    """
    Registers DejaVu fonts for ReportLab, Pillow, and Matplotlib.
    Returns: a dictionary with font names and full paths
    """
    if font_dir is None:
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    font_files = [
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans-Oblique.ttf",
        "DejaVuSans-BoldOblique.ttf",
        "DejaVuSansCondensed.ttf",
        "DejaVuSansCondensed-Bold.ttf",
        "DejaVuSansCondensed-Oblique.ttf",
        "DejaVuSansCondensed-BoldOblique.ttf",
        "DejaVuSansMono.ttf",
        "DejaVuSansMono-Bold.ttf",
        "DejaVuSansMono-Oblique.ttf",
        "DejaVuSansMono-BoldOblique.ttf",
        "DejaVuSans-ExtraLight.ttf",
    ]

    registered_fonts = {}

    for font_file in font_files:
        font_path = os.path.join(font_dir, font_file)
        font_name = os.path.splitext(font_file)[0]

        if not os.path.exists(font_path):
            print(f"⚠️ Missing: {font_file}")
            continue

        # Register for ReportLab
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
        except Exception as e:
            print(f"❌ ReportLab register failed for {font_name}: {e}")

        # Register for Matplotlib (append to font manager)
        try:
            fm.fontManager.addfont(font_path)
        except Exception as e:
            print(f"❌ Matplotlib register failed for {font_name}: {e}")

        registered_fonts[font_name] = font_path

    return registered_fonts


def get_pillow_font(font_name, size=16, font_map=None):
    """
    Load a TrueType font for use in Pillow.
    font_map is returned from register_dejavu_fonts()
    """
    if font_map and font_name in font_map:
        try:
            return ImageFont.truetype(font_map[font_name], size)
        except Exception as e:
            print(f"❌ Pillow failed to load {font_name}: {e}")
    else:
        print(f"⚠️ Font '{font_name}' not found.")
    return None
