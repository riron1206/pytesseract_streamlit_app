"""
Pythonで画像データを読み込み、OCR(光学的文字認識)でテキストデータに変換
tesseract: Google TesseractのPythonラッパー
pyocr: PythonからOCRエンジンを使うためのラッパー

参考:
- https://rightcode.co.jp/blog/information-technology/python-tesseract-image-processing-ocr

Usage:
    $ conda activate lightning
    $ streamlit run ./app.py
"""

# ==============================================
# app.py
# ==============================================
import re
import sys
import cv2
import argparse
import numpy as np
import pandas as pd
import traceback
import streamlit as st
from PIL import Image

import pytesseract
from pytesseract import Output

import pyocr
import pyocr.builders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="image_to_string")
    parser.add_argument("--lang", type=str, default="jpn")
    parser.add_argument("--psm", type=str, default="3")
    return parser.parse_args()


def load_file_up_image(file_up, args):
    pillow_img = Image.open(file_up).convert("RGB")
    # 一定の明るさを持つピクセルは白にする」という前処理
    # https://rightcode.co.jp/blog/information-technology/python-tesseract-image-processing-ocr
    size = pillow_img.size
    img2 = Image.new("RGB", size)
    border = args.border
    for x in range(size[0]):
        for y in range(size[1]):
            r, g, b = pillow_img.getpixel((x, y))
            if r > border or g > border or b > border:
                r = 255
                g = 255
                b = 255
            img2.putpixel((x, y), (r, g, b))
    return img2


def run_pyocr(img, args):
    """pyocrでOCR"""
    # pyocrへ利用するOCRエンジンをTesseractに指定する。
    tools = pyocr.get_available_tools()
    tool = tools[0]
    if args.mode == "image_to_string":
        # 画像から文字を読み込む
        ocr = tool.image_to_string(
            img,
            lang=args.lang,
            builder=pyocr.builders.TextBuilder(tesseract_layout=int(args.psm)),
        )
    return ocr


def pil2cv(image):
    """PIL型 -> OpenCV型
    https://qiita.com/derodero24/items/f22c22b22451609908ee"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


class Tesseract:
    """tesseractでOCR"""

    def __init__(self, args):
        self.args = args

    def image_to_string(self, img):
        """画像から文字を読み込む"""
        ocr_str = pytesseract.image_to_string(
            img, lang=self.args.lang, config=f"--psm {self.args.psm}"
        )
        # print(ocr_str)
        return ocr_str

    def image_to_osd(self, img):
        """画像から文字方向を読み込む"""
        ocr_osd = pytesseract.image_to_osd(
            img,
            output_type=pytesseract.Output.DICT,
        )
        return ocr_osd

    def image_to_boxes(self, img):
        """
        画像から文字とその位置を読み込む
        文字、左下座標(x, y)、右上座標(x, y)、ページ番号という形式を返す
        https://data.gunosy.io/entry/ocr-tesseract-deeplearning
        """

        def put_bbox(image, results):
            """
            検出結果をバウンディングボックスで描画
            """
            # 個々のテキストのローカライズをループ
            for i in range(0, len(results["text"])):
                # 現在の結果からテキスト領域のバウンディングボックス座標を抽出
                x = results["left"][i]
                y = results["top"][i]
                w = results["width"][i]
                h = results["height"][i]

                # OCRテキスト自体を、テキストの地域化の信頼性と一緒に抽出
                text = results["text"][i]
                conf = int(results["conf"][i])

                # 弱い信頼度のテキストのローカライズをフィルタリング
                if conf > 0.0:
                    # 信頼度とテキストを端末に表示
                    # print("Confidence: {}".format(conf))
                    # print("Text: {}".format(text))
                    # print("")

                    # 非 ASCII 文字を削除して，OpenCV を用いて画像上にテキストを描画し，テキストの周囲にテキストと一緒に外接枠を描画
                    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,)

        img = pil2cv(img)
        ocr = pytesseract.image_to_data(
            img,
            lang=self.args.lang,
            output_type=Output.DICT,
            config=f"--psm {self.args.psm}",
        )
        put_bbox(img, ocr)

        ocr_text = [" " if s == "" else s for s in ocr["text"]]
        ocr_text = "".join(ocr_text)
        ocr_text = re.sub("  +", "\n\n", ocr_text)
        # ocr_text.replace("   ", "\n")
        # print(ocr_text)

        return img, ocr_text


def main():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.title("Simple OCR App")
    st.write("")

    args = get_args()

    # サイドバー ラジオボタン
    st_mode = st.sidebar.radio(
        "Select Mode",
        (
            "image_to_string",
            "image_to_boxes",
            "image_to_osd",
        ),
    )
    args.__setattr__("mode", st_mode)

    if st_mode in ["image_to_string", "image_to_boxes"]:
        st_lang = st.sidebar.radio(
            "Select Lang",
            (
                "jpn",
                "eng",
                "eng+jpn",
            ),
        )
        args.__setattr__("lang", st_lang)

    if st_mode in ["image_to_string", "image_to_boxes"]:
        st_psm = st.sidebar.radio(
            "Select config psm",
            (
                "3",
                "6",
            ),
        )
        args.__setattr__("psm", st_psm)

    st_border = st.sidebar.slider("rgbのしきい値", 50, 255, step=10, value=100)
    args.__setattr__("border", st_border)

    # Tesseractインスタンス
    tesseract_cls = Tesseract(args)

    # ファイルupload
    file_up = st.file_uploader("Upload an image for OCR", type=["png", "jpg", "jpeg"])

    if file_up is not None:
        img = load_file_up_image(file_up, args)

        ocr = None
        try:
            if args.mode == "image_to_string":
                ocr = tesseract_cls.image_to_string(img)
                # ocr = run_pyocr(img, args)
            elif args.mode == "image_to_osd":
                ocr = tesseract_cls.image_to_osd(img)
            elif args.mode == "image_to_boxes":
                img, ocr = tesseract_cls.image_to_boxes(img)
        except Exception as e:
            traceback.print_exc()

        st.write("## Load Image")
        st.image(img)
        st.write("## OCR")
        st.write(ocr)
    else:
        img_url = "https://www.city.kagoshima.lg.jp/shimin/shiminbunka/shimin/images/mynumcard_back.png"
        st.image(
            img_url,
            caption="Sample Image. Please download and upload.",
            use_column_width=True,
        )


if __name__ == "__main__":
    main()
