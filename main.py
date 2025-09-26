import torch

# ? why did original tf version do this?
def make_divisible(v:float, divisor:int, min_value:int):
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if (new_v < 0.9 * v):
        return new_v + divisor
    return new_v

class SqueezeAndExcitation(torch.nn.Module):
    def __init__(self, in_channels):
        self.squeeze = make_divisible(in_channels // 4, 8, 8)
        super().__init__()
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.f1 = torch.nn.Conv2d(in_channels, self.squeeze, 1)
        self.act = torch.nn.ReLU()
        self.f2 = torch.nn.Conv2d(self.squeeze, in_channels, 1)
        self.scale_act = torch.nn.Hardsigmoid()

    def forward(self, input:torch.Tensor):
        output = self.avg(input)
        output = self.f1(output)
        output = self.act(output)
        output = self.f2(output)
        return self.scale_act(output) * input

class PrintShape(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer_num = layer

    def forward(self, input:torch.Tensor):
        print(f"Layer: {self.layer_num}, Shape: {input.shape}")
        return input


class Bottleneck(torch.nn.Module):
    def __init__(self, kernel_size, in_channels, exp_channels, out_channels, activation, stride, se, layer):
        super().__init__()
        selayer = torch.nn.Identity()
        if se:
            selayer = SqueezeAndExcitation(exp_channels)
        pad = 1
        if kernel_size == 5:
            pad = 2

        expand = torch.nn.Identity()
        if exp_channels != in_channels:
            expand = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=exp_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(num_features=exp_channels),
                activation(),
            )

        self.bottleneck = torch.nn.Sequential(
            # Expand channels
            expand,
            # 3x3 depthwise convolution with activation
            torch.nn.Conv2d(in_channels=exp_channels,
                            out_channels=exp_channels,
                            groups=exp_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=pad,
                            bias=False),

            torch.nn.BatchNorm2d(num_features=exp_channels),
            activation(),
            # squeeze and excitation
            selayer,
            # Linear pointwise convolution
            torch.nn.Conv2d(in_channels=exp_channels, out_channels=out_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),

            # debug
            PrintShape(layer)
        )

    def forward(self, input:torch.Tensor):
        output = self.bottleneck(input)
        if (output.shape == input.shape):
            return output + input
        else:
            return output


class MobileNetV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Hardswish(),
            PrintShape(0)
        )
        self.bottlenecks = torch.nn.Sequential(
            Bottleneck(3, 16, 16, 16, torch.nn.ReLU, 1, False ,1),
            Bottleneck(3, 16, 64, 24, torch.nn.ReLU, 2, False, 2),
            Bottleneck(3, 24, 72, 24, torch.nn.ReLU, 1, False, 3),
            Bottleneck(5, 24, 72, 40, torch.nn.ReLU, 2, True, 4),
            Bottleneck(5, 40, 120, 40, torch.nn.ReLU, 1, True, 5),
            Bottleneck(5, 40, 120, 40, torch.nn.ReLU, 1, True, 6),
            Bottleneck(3, 40, 240, 80, torch.nn.Hardswish, 2, False, 7),
            Bottleneck(3, 80, 200, 80, torch.nn.Hardswish, 1, False, 8),
            Bottleneck(3, 80, 184, 80, torch.nn.Hardswish, 1, False, 9),
            Bottleneck(3, 80, 184, 80, torch.nn.Hardswish, 1, False, 10),
            Bottleneck(3, 80, 480, 112, torch.nn.Hardswish, 1, True, 11),
            Bottleneck(3, 112, 672, 112, torch.nn.Hardswish, 1, True, 12),
            Bottleneck(5, 112, 672, 160, torch.nn.Hardswish, 2, True, 13),
            Bottleneck(5, 160, 960, 160, torch.nn.Hardswish, 1, True, 14),
            Bottleneck(5, 160, 960, 160, torch.nn.Hardswish, 1, True, 15),

            torch.nn.Conv2d(160, 960, 1, bias=False),
            torch.nn.BatchNorm2d(960),
            torch.nn.Hardswish(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(960, 1280, 1),
            torch.nn.Hardswish(),
            torch.nn.Conv2d(1280, 1000, 1)
        )

    def forward(self, input:torch.Tensor):
        output = self.layer0(input)
        output = self.bottlenecks(output)
        output = self.classifier(output)
        return output.reshape(1, -1)

if __name__ == "__main__":
    input = torch.randn((1, 3, 224, 224))
    model = MobileNetV3()
    output = model(input)
    print(output.shape)
