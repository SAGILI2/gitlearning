import cv2
import imutils
import os
import sys
from pathlib import Path
from flask import Flask, redirect, url_for, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename


from main.unet_model import get_mask
from main.orientation_model import get_orientation
from main.start_ocr import model_extract
from utils.text_detection import TextDetection


# FOLDERS DECLARATION
project_folder = Path(__file__).parent
project_parent_folder = project_folder.parent
uploads_folder = project_parent_folder / "OCR_UPLOADS"
modified_jpg_folder = project_parent_folder / "MOD_JPG"


def create_folders():
    if not uploads_folder.exists():
        uploads_folder.mkdir(parents=True)
    if not modified_jpg_folder.exists():
        modified_jpg_folder.mkdir(parents=True)


create_folders()


sys.path.append(os.getcwd())

# Define a flask app
app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

basepath = os.getcwd()
host_address = "0.0.0.0"
port = 8113


## read config file for URLs hosting OCR
parallel_processors_ = []
print("GORETEXT")
try:
    with open("/home/ubuntu/PLATFORM_OCR_API/PARALLEL_PROCESSING", "r") as fp:
        parallel_ = fp.readlines()

    for elem in parallel_:
        parallel_processors_.append(elem.strip("\n"))
        print("XXXXXXXXXXXFOUND URLS", parallel_processors_)

except:
    print("YYYYYYYYYYYYYYYYPool not complete..use sequential processing")
    parallel_processors_ = []


@app.route("/td_api", methods=["GET", "POST"])
@cross_origin()
def start_td():
    if request.method == "POST":
        # Get the file from post request
        try:
            im_path = request.form["orig_path"]
            dst_path = request.form["dst_path"]
            img = cv2.imread(im_path)
            im_ = cv2.imread(im_path, 0)
            orig_img = img.copy()
            text_mask = get_mask([im_])[0]
            text_mask = cv2.threshold(text_mask, 125, 255, cv2.THRESH_BINARY)[1]
            text_mask_bw = text_mask
            text_mask = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
            orientation_pred = get_orientation([text_mask])[0]
            if orientation_pred == 90:
                img = imutils.rotate_bound(img, -90)
                text_mask_bw = imutils.rotate_bound(text_mask_bw, -90)
            ocr_res = TextDetection(img, im_path, text_mask_bw)
            im_out = ocr_res.oriented_orig_img_color.copy()
            color_1 = [0, 0, 255]
            color_2 = [0, 255, 0]
            color_3 = [255, 0, 0]
            curr_color = color_1
            lines = ocr_res.lines
            if True:
                for line_idx, line in enumerate(lines):
                    if curr_color == color_1:
                        curr_color = color_2
                    elif curr_color == color_2:
                        curr_color = color_3
                    else:
                        curr_color = color_1
                    pts_line = []
                    for block_idx, block in enumerate(line):
                        # id = block["id"]
                        [x, y, w, h] = block["pts"]
                        new_id = block["id"]
                        block["id"] = new_id
                        lines[line_idx][block_idx]["id"] = new_id
                        # if h<13:
                        cv2.rectangle(im_out, (x, y), (x + w, y + h), curr_color, 2)
                        # remove_bg(im_out_2[y:y+h,x:x+w],h,w)
                        pts_line.append(block["pts"])
                        # print("w",w,"h",h)
                        # print('BLOCK->', block)
                        # text_blocks.append(block)
                    # print("pts", pts_line)
                cv2.imwrite(dst_path, im_out)
        except:
            print("Error in ", im_path)
        return jsonify("Success")


@app.route("/extractText", methods=["GET", "POST"])
@cross_origin()
def start_ocr():
    if request.method == "POST":

        # Get the file from post request
        files = request.files
        file = files["file"]
        name = secure_filename(file.filename)

        img_path = uploads_folder / name
        mod_img_path = modified_jpg_folder / name
        file.save(img_path)
        ocr_res = model_extract(str(img_path), str(mod_img_path))
        if ocr_res:
            mod_img_path.unlink(missing_ok=True)
            return jsonify(ocr_res)
        return jsonify({"line": [], "path": ""})


if __name__ == "__main__":
    # app.run(host=host_address, debug=False, port=port, threaded=False)
    print("starting on ", port)
    app.run(host=host_address, port=port, threaded=False)
