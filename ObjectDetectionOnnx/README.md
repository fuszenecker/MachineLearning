# Object detection with an ONNX model

Download the model `resnet152-v1-7.onnx` from [this page](https://github.com/onnx/models/tree/main/archive/vision/classification/resnet).

MD5:

```text
0a16b87d6c9e0e21528773def0410203  ./resnet152-v1-7.onnx
```

## Flatpak build

Generating `sources.json`:

```sh
python3 ./Tools/flatpak-dotnet-generator.py sources.json ./ObjectDetectionOnnx.csproj
```

Build:

```sh
flatpak-builder ../flatpak-app --force-clean ./net.fuszenecker.ObjectDetection.yaml
```