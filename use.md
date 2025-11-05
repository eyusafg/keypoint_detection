# 说明
## train
    python tools/keypoint/train/train.py --cfg 参数文件路径 

## test
    python tools/keypoint/test/single_test.py --cfg 参数文件路径 --load 模型路径 --img_dir 图像文件夹路径

## export onnx
    python tools/keypoint/export/export.py --cfg 参数文件路径 --load 模型路径 --output onnx输出路径

# 补全
    cd 在代码根目录
    pip install .      # 已安装syt_vision_flow可跳过，若修改代码的输入参数，需要重新安装一次
    source completion.sh 

### 按TAB补全 (参数同上)
    syt_train --model kpt或segm --cfg 参数文件路径 --load 预训练模型路径
    syt_test xxx
    syt_eval xxx
    sty_test xxx
    sty_export xxx
