import random
import base64
from io import BytesIO
from PIL import Image


from flask import current_app as app, g, Blueprint, request, jsonify, send_file
from hutil.ext.captcha import ImageCaptcha
from lotusland.models import CaptchaNet, CaptchaDSOD

bp = Blueprint("serving", __name__)


class CaptchaGenerator():

    def __init__(self, fonts):
        width = 128
        height = 48
        self.letters = "0123456789abcdefghijkmnopqrstuvwxyzABDEFGHJKMNRT"
        font_sizes = (28, 32, 36, 40, 44, 48)
        self.image = ImageCaptcha(
            width, height, fonts=fonts, font_sizes=font_sizes)

    def generate(self):
        chars = random.choices(self.letters, k=4)
        img = self.image.generate_image(
            chars, noise_dots=.3, noise_curve=.3, rotate=20)
        return img


def get_captcha_generator():
    if 'captcha_gen' not in g:
        g.captcha_gen = CaptchaGenerator(
            app.config["FONTS"]
        )
    return g.captcha_gen


def get_captcha_detection_model():
    if 'captcha_det' not in g:
        g.captcha_det = CaptchaDSOD(
            app.config["CAPTCHA_DSOD"],
        )
    return g.captcha_det


def get_captcha_model():
    if 'captcha_net' not in g:
        g.captcha_net = CaptchaNet(
            app.config["CAPTCHANET"]
        )
    return g.captcha_net


def init_app():
    g.captcha_net = CaptchaNet(
        app.config["CAPTCHANET"]
    )
    g.captcha_det = CaptchaDSOD(
        app.config["CAPTCHA_DSOD"],
    )
    g.captcha_gen = CaptchaGenerator(
        app.config["FONTS"]
    )


@bp.route("/api/captcha", methods=["POST"])
def captcha():
    f = request.files['img']
    # img = Image.open(f).convert('RGB')
    img = Image.open(f).convert('RGB')
    net = get_captcha_model()
    pred = net.predict(img)
    return pred


@bp.route("/api/captcha/generate", methods=["GET"])
def generate_captcha():
    gen = get_captcha_generator()
    img = gen.generate()
    img_io = BytesIO()
    img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@bp.route("/api/captcha/detection", methods=["POST"])
def captcha_detection():
    f = request.files['img']
    img = Image.open(f).convert('RGB')
    net = get_captcha_detection_model()
    img, dets = net.predict(img)

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = str(base64.b64encode(buffered.getvalue()), 'ascii')
    return jsonify({
        "img": img_str,
        "detections": dets,
    })
