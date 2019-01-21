from flask import current_app as app, g, Blueprint, request
from PIL import Image
from lotusland.models import CaptchaNet

bp = Blueprint("serving", __name__)


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


@bp.route("/api/captcha", methods=["POST"])
def captcha():
    f = request.files['img']
    # img = Image.open(f).convert('RGB')
    img = Image.open(f).convert('RGB')
    net = get_captcha_model()
    pred = net.predict(img)
    return pred
