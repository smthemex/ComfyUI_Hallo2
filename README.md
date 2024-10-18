# ComfyUI_Hallo2
[Hallo2](https://github.com/fudan-generative-vision/hallo2): Long-Duration and High-Resolution Audio-driven Portrait Image Animation,

1.Installation
-----
In the ./ComfyUI /custom_node directory, run the following:   
 ``` python 
 git clone https://github.com/smthemex/ComfyUI_Hallo2

 ```
2.requirements  
----
```
pip install -r requirements.txt
```
if using embeded comfyUI,in your "X:\ComfyUI_windows\python_embeded "(便携包的comfyUI用户在python_embeded目录下用以下命令安装)   
```
python -m pip install -r requirements.txt
```
Possible installation difficulties that may be encountered（可能会遇到的安装难题）：   
* 2.1  audio-separator   
* 2.1.1  If' pip install audio-separator' building wheel fail（diffq），makesure has install [visual-cpp-build-tools](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools ) in window   
安装audio-separator可能会出现vs的报错，确认你安装了[visual-cpp-build-tools](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools )    
* 2.1.2   Although there are ‘visual-cpp-build-tools’, it still fails（diffq）. If you are using the ComfyUI portable package or Akiba package, please add the interpreter address to the Windows system variable.    
虽然有‘visual-cpp-build-tools，但是还是失败（diffq），如果使用的是comfyUI便携包，或者秋叶包，请将解释器地址加入windows的系统变量里，Linux用户，你都用Linux了，就不用我教了吧，window的做法是，将X:\ComfyUI_windows\python_embeded 和F:\ComfyUI_windows\python_embeded\Scripts 2个地址加入Path系统变量里。   
* 3.2  ffmpeg   
* 3.3  If the module is missing, Remove the requirements' # symbol,please pip install       
少了啥，就去掉#号，重新安装    

3 checkpoints
----
所有模型下载地址（all checkpoints）：[huggingface](https://huggingface.co/fudan-generative-ai/hallo2/tree/main)

```
├── ComfyUI/models/hallo/
|-- audio_separator/
|   |-- download_checks.json
|   |-- mdx_model_data.json
|   |-- vr_model_data.json
|   `-- Kim_Vocal_2.onnx
|-- face_analysis/
|   `-- models/
|       |-- face_landmarker_v2_with_blendshapes.task  # face landmarker model from mediapipe
|       |-- 1k3d68.onnx
|       |-- 2d106det.onnx
|       |-- genderage.onnx
|       |-- glintr100.onnx
|       `-- scrfd_10g_bnkps.onnx
|-- facelib
|   |-- detection_mobilenet0.25_Final.pth
|   |-- detection_Resnet50_Final.pth
|   |-- parsing_parsenet.pth
|   |-- yolov5l-face.pth
|   `-- yolov5n-face.pth
|-- hallo2
|   |-- net_g.pth
|   `-- net.pth
|-- motion_module/
|   `-- mm_sd_v15_v2.ckpt
`-- wav2vec/
    `-- wav2vec2-base-960h/
        |-- config.json
        |-- feature_extractor_config.json
        |-- model.safetensors
        |-- preprocessor_config.json
        |-- special_tokens_map.json
        |-- tokenizer_config.json
        `-- vocab.json
```
Normal checkpoints   
```
├── ComfyUI/models/
|-- upscale_models/
|   `-- RealESRGAN_x2plus.pth
|-- vae/
|   `-- vae-ft-mse-840000-ema-pruned.safetensors
|-- checkpoints/
|   `-- v1-5-pruned-emaonly.safetensors # any sd1.5
       
```
5 Example
----     


6 Citation
------
hallo2
```
@misc{cui2024hallo2,
	title={Hallo2: Long-Duration and High-Resolution Audio-driven Portrait Image Animation},
	author={Jiahao Cui and Hui Li and Yao Yao and Hao Zhu and Hanlin Shang and Kaihui Cheng and Hang Zhou and Siyu Zhu and️ Jingdong Wang},
	year={2024},
	eprint={2410.07718},
	archivePrefix={arXiv},
	primaryClass={cs.CV}
}
```

