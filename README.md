# pytorch-cuda
PyTorch example to consume CUDA input data directly.

## Training

Training is done by python code inside folder training. The dataset used is Fashion MNIST. With a successful training, two model files will be created, model.pth and model.pt. The model.pt file is the Torch Script Model which can be loaded in libtorch.

## libtorch inference

An example of libtorch inference is inside folder libtorch-infer. It includes a piece of hard code input data extracted from the the Fashion MNIST test data set indexed with 100. The model.pt file needs to be in the working directory when running.

Set Torch_DIR to point to torch cmake directory.
