set(CMAKE_CXX_FLAGS "-fPIC")

./ArmnnConverter  --model-format onnx-binary --model-path ./model/model.onnx --input-name input --output-name output --output-path ./model/model.armnn