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
sh script/install.sh
conda activate 3dvg
```

## Run
Directly run
```
sh train.sh
```
or
```
CUDA_VISIBLE_DEVICES='0' python svg_render.py x=dream3dvg seed=1 x.style='sketch' x.num_paths=32 "prompt='A benz car'" result_path='./workspace/3dvg/Benz' "x.camera_param.init_prompt='A benz car'"
CUDA_VISIBLE_DEVICES='0' python svg_render.py x=dream3dvg seed=1 x.style='iconography' x.num_paths=128"prompt='A carb'" result_path='./workspace/3dvg/Carb' "x.camera_param.init_prompt='A carb'"
```
