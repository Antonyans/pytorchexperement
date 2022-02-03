import os
import torch


class BaseFeature:
    def __init__(self, base_file, path, device):
        self.base_dir = os.path.abspath(os.path.dirname(base_file))
        self.model_path = os.path.join(self.base_dir, 'weights', path)

        self.model = None
        self._load_model(self.model_path)
        self._device = self.set_device(device)

        self.transform = self._create_transform()

    @property
    def _margin(self):
        return 0.1

    @property
    def device(self):
        return self._device

    def set_device(self, device):
        device = torch.device(device)
        self.model.to(device)
        self._device = device
        return device

    def _load_model(self, path):
        raise NotImplementedError

    def _create_transform(self):
        raise NotImplementedError

    def _img_preprocess(self, img_list):
        if img_list is None:
            return []

        images = []
        for img in img_list:
            img = self.transform(img)
            images.append(img)
        images = torch.stack(images)
        images = images.to(self.device)
        return images

    def extract_faces(self, facedet_response):
        if isinstance(facedet_response, dict):
            facedet_response = [facedet_response]
        faces = []
        print('*'*60)
        print(facedet_response)
        print('*'*60)
        for response in facedet_response:
            image = response["_meta"]["image"]
            rect = response["rectangle"]
            box = [rect["left"], rect["top"], rect["right"], rect["bottom"]]
            margin_width = ((box[2] - box[0]) * self._margin)
            margin_height = ((box[3] - box[1]) * self._margin)
            margin = [margin_width, margin_height]

            box = [round(max(box[0] - margin[0] / 2, 0)),
                   round(max(box[1] - margin[1] / 2, 0)),
                   round(min(box[2] + margin[0] / 2, image.size[0])),
                   round(min(box[3] + margin[1] / 2, image.size[1]))]

            faces.append(image.crop(box))

        return faces

    def __call__(self, facedet_response):
        images = self.extract_faces(facedet_response)
        img = self._img_preprocess(images)
        outputs = self.model(img).cpu().detach()
        response = self._output_postprocess(outputs)
        return response

    def _output_postprocess(self, outputs):
        raise NotImplementedError
