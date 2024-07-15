from utils.box_ops import box_xyxy_to_cxcywh


class NormalizeBoxes(object):
    def __call__(self, image, target=None):
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"].float()
            boxes[..., 0::2] /= w
            boxes[..., 1::2] /= h
            target["boxes"] = box_xyxy_to_cxcywh(boxes)
        return image, target
