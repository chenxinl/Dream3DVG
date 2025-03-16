#  Implementation of "Empowering Vector Graphics with Consistently Arbitrary Viewing and View-dependent Visibility"

![](/assets/teaster.png)

## Setup
```
git clone https://github.com/chenxinl/Dream3DVG.git
cd Dream3DVG
```
## Environment
To set up the environment, please run:
```
chmod +x script/install.sh
sh script/install.sh
conda activate 3dvg
```
Remind to check the `cuda version` first (use `nvcc -V`), and modify the pytorch installation command in `install.sh`.

If you meet the `gcc error` when building or running `pydiffvg`, try to update `gcc` using:
```
conda install -c conda-forge gcc
```
Remind to modify the GPU arch in `L48` of `install.sh`, take `NVIDIA A100-sm_80` for example:
```
sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode=arch=compute_80,code=sm_80")/' CMakeLists.txt
```

## Run
Directly run
```
sh train.sh
```
or
```
CUDA_VISIBLE_DEVICES='0' python svg_render.py x=dream3dvg seed=1 x.style='sketch' x.num_paths=32 "prompt='A benz car'" result_path='./workspace/3dvg/Benz' "x.camera_param.init_prompt='A benz car'"
CUDA_VISIBLE_DEVICES='0' python svg_render.py x=dream3dvg seed=1 x.style='iconography' x.num_paths=128 "prompt='A carb'" result_path='./workspace/3dvg/Carb' "x.camera_param.init_prompt='A carb'"
```

## Acknowledgments
Thanks for the awesome repository:

`DiffVG`: https://github.com/BachiLi/diffvg

`Pytorch-SVGRenderer`: https://github.com/ximinng/PyTorch-SVGRender

`LucidDreamer`: https://github.com/EnVision-Research/LucidDreamer

`3Doodle`: https://github.com/changwoonchoi/3Doodle
## Citation
