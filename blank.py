import argparse
import time
import torch
import logging
import cv2
import sys

sys.path.append('yolov5')
from yolov5.utils.datasets import LoadImages
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.models.experimental import attempt_load


class Model:
    """ Class to interact with pytorch model (including YOLOv5) """
    def __init__(self, weights_path, img_size=640, device='', conf_thres=0.5, iou_thres=0.5, classes=None):
        self._weights_path = weights_path
        self._img_size = img_size
        self._device = select_device(device)
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._classes = classes

        self._half = self._device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self._model = attempt_load(self._weights_path, map_location=self._device)  # load FP32 model
        self._img_size = check_img_size(self._img_size, s=self._model.stride.max())  # check img_size

        if self._half:
            self._model.half()  # to FP16

    @property
    def names(self):
        return self._model.module.names if hasattr(self._model, 'module') else self._model.names

    def predict(self, img, im0s):
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self._model(img)[0]

        # Apply NMS
        det = non_max_suppression(pred, self._conf_thres, self._iou_thres, classes=self._classes)[0]
        t2 = time_synchronized()
        logging.info('Inference time: {:.3f}s'.format(t2 - t1))

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            return det


def main(input_path, save_path, weights_path, device):
    start_time = time.time()
    model = Model(weights_path, device=device, classes=[0])

    dataset = LoadImages(input_path, img_size=640)

    fps = dataset.cap.get(cv2.CAP_PROP_FPS)
    w = int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, (w, h))

    for frame_number, (path, img, im0s, vid_cap) in enumerate(dataset, 1):
        img_to_draw = im0s.copy()
        det = model.predict(img, im0s)
        print(f'detected {len(det[:, -1])} people')

        # ==================================
        # todo: add your tracking somewhere here
        # ==================================

        if det is not None:
            for *xyxy, conf, cls_id in det:
                # plotting bboxes of detected people
                plot_one_box(xyxy, img_to_draw, color=(0, 0, 255), line_thickness=2,
                             label=model.names[int(cls_id)])  # todo: add person id to label

        vid_writer.write(img_to_draw)

    vid_writer.release()

    print('Finished in {:.3f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='vtest.avi', help='source file')
    parser.add_argument('--save-path', type=str, default='result.avi', help='output filename')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights-path', default='yolov5s.pt', help='path to .pt file with model weights')

    args = parser.parse_args()

    with torch.no_grad():
        main(input_path=args.input_path, save_path=args.save_path, weights_path=args.weights_path, device=args.device)
