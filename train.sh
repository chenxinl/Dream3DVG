export CUDA_VISIBLE_DEVICES='0'

# sketch
style='sketch'
num_paths=32

python svg_render.py x=dream3dvg seed=2 x.style=$style x.num_paths=$num_paths "prompt='A flamingo'" result_path='./workspace/3dvg/Flamingo' "x.camera_param.init_prompt='A flamingo'"
python svg_render.py x=dream3dvg seed=1 x.style=$style x.num_paths=$num_paths "prompt='A benz car'" result_path='./workspace/3dvg/Benz' "x.camera_param.init_prompt='A benz car'"

# iconography
style='iconography'
num_paths=128

python svg_render.py x=dream3dvg seed=1 x.style=$style x.num_paths=$num_paths "prompt='A carb'" result_path='./workspace/3dvg/Carb' "x.camera_param.init_prompt='A carb'"
python svg_render.py x=dream3dvg seed=1 x.style=$style x.num_paths=$num_paths "prompt='A yellow school bus'" result_path='./workspace/3dvg/Bus' "x.camera_param.init_prompt='A yellow school bus'"

# set conf/config.yaml diffusers.download: True when first run
