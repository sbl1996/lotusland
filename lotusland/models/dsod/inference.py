
import heapq
from PIL import Image, ImageDraw

import torch

from torchvision.transforms import ToTensor, Resize, Compose

from hutil.inference import freeze, clip
from hutil.detection import BBox, iou_11, transform_bbox, scale_box, draw_bboxes

from lotusland.models.dsod.ssd import SSDInference, compute_default_boxes, compute_scales
from lotusland.models.dsod.model import DSOD


def filter_dets(dets):
    def t(d):
        return transform_bbox(
            d['bbox'],
            BBox.LTWH,
            BBox.LTRB
        )
    ret = [dets[0]]
    for d in dets[1:]:
        if iou_11(t(d), t(ret[-1])) > 0.4:
            if d['confidence'] > ret[-1]['confidence']:
                ret[-1] = d
        else:
            ret.append(d)
    return ret


class CaptchaDSOD:

    def __init__(self, model_path):
        self.letters = "0123456789abcdefghijkmnopqrstuvwxyzABDEFGHJKMNRT"
        NUM_CLASSES = len(self.letters) + 1
        self.width = 128
        self.height = 48
        LOCATIONS = [
            (8, 3),
            (4, 2),
        ]
        ASPECT_RATIOS = [
            (1, 2, 1/2),
            (1, 2, 1/2),
        ]
        ASPECT_RATIOS = [torch.tensor(ars) for ars in ASPECT_RATIOS]
        NUM_FEATURE_MAPS = len(ASPECT_RATIOS)
        SCALES = compute_scales(NUM_FEATURE_MAPS, 0.2, 0.9)
        DEFAULT_BOXES = [
            compute_default_boxes(lx, ly, scale, ars)
            for (lx, ly), scale, ars in zip(LOCATIONS, SCALES, ASPECT_RATIOS)
        ]

        self.img_transform = Resize((self.height, self.width))
        self.tensor_transform = ToTensor()

        out_channels = [
            (NUM_CLASSES + 4) * len(ars)
            for ars in ASPECT_RATIOS
        ]
        net = DSOD([3, 4, 4, 4], 36, out_channels=out_channels, reduction=1)
        net.load_state_dict(torch.load(
            model_path, map_location='cpu')["model"])

        net.eval()
        freeze(net)
        clip(net)

        self.net = net

        self.inference = SSDInference(
            width=self.width, height=self.height,
            f_default_boxes=DEFAULT_BOXES,
            num_classes=NUM_CLASSES,
        )

    def predict(self, img):
        img = self.img_transform(img)
        x = self.tensor_transform(img)
        fs = self.net(x[None])[0]
        detections = self.inference(fs)
        detections = heapq.nlargest(10, detections, key=lambda d: d.confidence)
        dets = [
            {
                "bbox": scale_box(
                    transform_bbox(d.box,
                                   format=BBox.LTRB,
                                   to=BBox.LTWH),
                    src_size=(self.width, self.height),
                    dst_size=img.size),
                "label": self.letters[d.class_id],
                "confidence": d.confidence
            }
            for d in detections
        ]

        dets = filter_dets(
            sorted(dets, key=lambda d: (d['bbox'][0], -d['confidence'])))
        draw_bboxes(img, dets)

        return img, dets


def draw_bboxes(img, anns):
    draw = ImageDraw.Draw(img)
    for ann in anns:
        bbox = transform_bbox(
            ann["bbox"],
            format=BBox.LTWH,
            to=BBox.LTRB
        )
        draw.rectangle(bbox, outline='red', width=1)
        draw.text(bbox[:2], ann["label"], fill='black')
