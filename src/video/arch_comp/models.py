import torch
import torch.nn as nn
from model_zoo.i3d import InceptionI3dWrapper
from model_zoo.slowfast import SlowFast50, SlowFast152
from model_zoo.timeception import TimeceptionWrapper


class ModelSelect(nn.Module):
    def __init__(self, nb_frames, output_size, model_type):
        super(ModelSelect, self).__init__()
        self.model_type = model_type

        if model_type == "i3d":
            self.model = InceptionI3dWrapper(nb_frames=nb_frames, output_size=output_size)
        elif model_type == "timeception50":
            self.model = TimeceptionWrapper(
                nb_frames=nb_frames, base_model="resnet50", output_size=output_size
            )
        elif model_type == "timeception152":
            self.model = TimeceptionWrapper(
                nb_frames=nb_frames, base_model="resnet152", output_size=output_size
            )
        elif model_type == "slowfast50":
            self.model = SlowFast50(class_num=output_size)
        elif model_type == "slowfast152":
            self.model = SlowFast152(class_num=output_size)
        else:
            raise Exception("unknown model type: " + model_type)

    def forward(self, x):
        """
        # input: [time, length, width, channels]
        # i3d: [batch, channels, time, length, width] -> [batch, label, 3]
        # mobilenet: [time, length, width, channels] -> [1, label]
        # timeception: [batch, time, channels, length, width] -> [batch, label]
        # slowfast: [batch, channels, time, length, width] -> [batch, label]
        """

        if (
            self.model_type == "i3d" or self.model_type == "slowfast50" or
            self.model_type == "slowfast152"
        ):
            x = x.permute([0, 4, 1, 2, 3])
        elif self.model_type == "timeception50" or self.model_type == "timeception152":
            x = x.squeeze(0)
            x = x.permute([0, 3, 1, 2])
        else:
            raise Exception("unknown model type: " + self.model_type)

        y = self.model(x)
        y = y.view(1, -1)

        return y


if __name__ == "__main__":
    nb_frames = 128
    x = torch.ones([1, nb_frames, 224, 224, 3])

    model = ModelSelect(nb_frames=nb_frames, output_size=100, model_type="i3d")
    y = model.forward(x)
