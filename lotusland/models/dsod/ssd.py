import torch

from hutil.detection import BBox, transform_bboxes, iou_1m, transform_bbox, non_max_suppression


def compute_default_boxes(lx, ly, scale, ars):
    default_boxes = torch.zeros(lx, ly, len(ars), 4)
    default_boxes[:, :, :, 0] = (torch.arange(
        lx, dtype=torch.float).view(lx, 1, 1).expand(lx, ly, len(ars)) + 0.5) / lx
    default_boxes[:, :, :, 1] = (torch.arange(
        ly, dtype=torch.float).view(1, ly, 1).expand(lx, ly, len(ars)) + 0.5) / ly
    default_boxes[:, :, :, 2] = scale * torch.sqrt(ars)
    default_boxes[:, :, :, 3] = scale / torch.sqrt(ars)
    return default_boxes


def compute_scales(num_feature_maps, s_min, s_max):
    return [
        s_min + (s_max - s_min) * k / (num_feature_maps - 1)
        for k in range(num_feature_maps)
    ]


def compute_loc_target(gt_box, default_boxes):
    box_txty = (gt_box[:2] - default_boxes[..., :2]) \
        / default_boxes[..., 2:]
    box_twth = torch.log(gt_box[2:] / default_boxes[..., 2:])
    return torch.cat((box_txty, box_twth), dim=-1)


class SSDInference:

    def __init__(self, width, height, f_default_boxes, num_classes, confidence_threshold=0.3, max_boxes=10, iou_threshold=0.45):
        self.width = width
        self.height = height
        self.f_default_boxes = f_default_boxes
        self.confidence_threshold = confidence_threshold
        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes

    def __call__(self, fs):
        detections = []
        for f, default_boxes in zip(fs, self.f_default_boxes):
            batch_size = f.size(0)
            lx, ly, num_ars = default_boxes.size()[:3]
            f = f.view(batch_size, lx, ly, num_ars, -1)
            boxes_txty = f[..., 0:2]
            boxes_twth = f[..., 2:4]
            logits = f[..., 4:]

            boxes_cxcy = boxes_txty.mul_(
                default_boxes[..., 2:]).add_(default_boxes[..., :2])
            boxes_wh = boxes_twth.exp_().mul_(default_boxes[..., 2:])
            boxes = f[..., :4]  # inplace
            boxes[..., [0, 2]] *= self.width
            boxes[..., [1, 3]] *= self.height
            boxes = transform_bboxes(
                boxes, format=BBox.XYWH, to=BBox.LTRB, inplace=True)
            confidences = torch.softmax(logits, dim=-1)

            mask = confidences > self.confidence_threshold

            for i in range(batch_size):
                for c in range(self.num_classes - 1):
                    bc_mask = mask[i, ..., c]
                    bc_confidences = confidences[i, ..., c][bc_mask]
                    bc_boxes = boxes[i][bc_mask]
                    indices = non_max_suppression(
                        bc_boxes, bc_confidences, self.iou_threshold)
                    for ind in indices:
                        detections.append(
                            BBox(
                                image_name=i,
                                class_id=c,
                                box=bc_boxes[ind].tolist(),
                                confidence=bc_confidences[ind].item(),
                                box_format=BBox.LTRB,
                            )
                        )
        return detections
