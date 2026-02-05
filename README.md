# condenser.cpp

A minimal, lightweight fork of [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) focused on text-to-image and image editing with FLUX.2 Klein.

**Status:** Work in progress

## Supported Models

- FLUX.2 Klein (4GB and 9GB variants)

## Build

```shell
git clone --recursive https://github.com/jcluts/condenser.cpp
cd condenser.cpp
mkdir build && cd build
```

### CPU

```shell
cmake ..
cmake --build . --config Release
```

### CUDA

```shell
cmake .. -DSD_CUDA=ON
cmake --build . --config Release
```

### Vulkan

```shell
cmake .. -DSD_VULKAN=ON
cmake --build . --config Release
```

### Metal (macOS)

```shell
cmake .. -DSD_METAL=ON
cmake --build . --config Release
```

## Credits

Based on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) by leejet.
