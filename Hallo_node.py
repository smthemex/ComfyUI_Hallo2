# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import io
import logging
import os
import random
from PIL import Image
import torch
import gc
import numpy as np
import torchaudio
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
import platform
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download

from .scripts.inference_long import cut_audio,Net,process_audio_emb,inference_process
from .hallo.datasets.audio_processor import AudioProcessor
from .hallo.models.unet_2d_condition import UNet2DConditionModel
from .hallo.datasets.image_processor import ImageProcessor
from .hallo.models.unet_3d import UNet3DConditionModel
from .hallo.models.face_locator import FaceLocator
from .hallo.models.image_proj import ImageProjModel
from .hallo.models.audio_proj import AudioProjModel
from .hallo.animate.face_animate import FaceAnimatePipeline
from .hallo.video_sr import run_realesrgan,pre_u_loader

from .utils import load_images,tensor2cv
import folder_paths
from comfy.utils import common_upscale

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))
# add checkpoints dir
weigths_current_path = os.path.join(folder_paths.models_dir, "Hallo")
if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)
    
try:
   folder_paths.add_model_folder_path("Hallo", weigths_current_path, False)
except:
    try:
        folder_paths.add_model_folder_path("Hallo", weigths_current_path)
        logging.warning("old comfyUI version")
    except:
        raise "please update your comfyUI version"
    
weigths_audio_path= os.path.join(weigths_current_path, "audio_separator")
if not os.path.exists(weigths_audio_path):
    os.makedirs(weigths_audio_path)
    
weigths_wav2vec_path= os.path.join(weigths_current_path, "wav2vec/wav2vec2-base-960h")
if not os.path.exists(weigths_wav2vec_path):
    os.makedirs(weigths_wav2vec_path)
    
weigths_motion_path= os.path.join(weigths_current_path, "motion_module")
if not os.path.exists(weigths_motion_path):
    os.makedirs(weigths_motion_path)
    
weigths_facelib_path= os.path.join(weigths_current_path, "facelib")
if not os.path.exists(weigths_facelib_path):
    os.makedirs(weigths_facelib_path)
    
weigths_face_analysis_path= os.path.join(weigths_current_path, "face_analysis/models")
weigths_face_analysis_dir= os.path.join(weigths_current_path, "face_analysis")
if not os.path.exists(weigths_face_analysis_path):
    os.makedirs(weigths_face_analysis_path)

weigths_hallo2_path= os.path.join(weigths_current_path, "hallo2")
if not os.path.exists(weigths_hallo2_path):
    os.makedirs(weigths_hallo2_path)

device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None and platform.system() in ['Linux', 'Darwin']:
    try:
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip()
            print(f"FFmpeg is installed at: {ffmpeg_path}")
        else:
            print("FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH.")
            print("For example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
    except Exception as e:
        pass

if ffmpeg_path is not None and ffmpeg_path not in os.getenv('PATH'):
    print("Adding FFMPEG_PATH to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pil_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil

def load_audio(use_cut,driving_audio_file,save_path,sample_rate,fps,wav2vec_model_path,wav2vec_only_last_features,audio_separator_model_file,clip_length):
    if use_cut:  # config.use_cut:
        audio_list = cut_audio(driving_audio_file,save_path)
        
        audio_emb_list = []
        l = 0
        
        audio_processor = AudioProcessor(
            sample_rate,
            fps,
            wav2vec_model_path,
            wav2vec_only_last_features,
            audio_separator_model_path=weigths_audio_path,
            audio_separator_model_name=audio_separator_model_file,
            cache_dir=
            os.path.join(save_path, "audio_preprocess")
        )
        
        for idx, audio_path in enumerate(audio_list):
            padding = (idx + 1) == len(audio_list)
            emb, length = audio_processor.preprocess(audio_path, clip_length,
                                                     padding=padding, processed_length=l)
            audio_emb_list.append(emb)
            l += length
        
        audio_emb = torch.cat(audio_emb_list)
        audio_length = l
    
    else:
        with AudioProcessor(
                sample_rate,
                fps,
                wav2vec_model_path,
                wav2vec_only_last_features,
                audio_separator_model_path=weigths_audio_path,
                audio_separator_model_name=audio_separator_model_file,
                cache_dir=os.path.join(save_path, "audio_preprocess")
        ) as audio_processor:
            audio_emb, audio_length = audio_processor.preprocess(driving_audio_file, clip_length)
            
    return audio_emb,audio_length

def load_unet(config,enable_zero_snr,base_model_path,motion_module_path,vae_path):
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if enable_zero_snr:  # config.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    
    config_path=os.path.join(current_node_path,"configs/unet/config.json")
    
    vae_config = os.path.join(current_node_path,"configs/vae")
    original_config=os.path.join(folder_paths.models_dir,"configs/v1-inference.yaml")
    try:
        vae = AutoencoderKL.from_single_file(vae_path, config=vae_config,original_config=original_config)
    except:
        try:
            vae = AutoencoderKL.from_single_file(vae_path, config=vae_config, original_config_file=original_config)
        except:
            raise "error"
            
    reference_unet = UNet2DConditionModel.from_pretrained_2d(
        base_model_path, subfolder=None,config_path=config_path,)
    
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        base_model_path,
        motion_module_path,
        subfolder=None,
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs),
        use_landmark=False,
        config_path=config_path,
    )
    # denoising_unet.set_attn_processor()
    return val_noise_scheduler,vae,reference_unet,denoising_unet

def pre_img(img_pil,save_path,face_expand_ratio,width,height,model_path):
    img_size = (width,height)
    clip_length = 16  # config.data.n_sample_frames
    with ImageProcessor(img_size, weigths_face_analysis_dir) as image_processor:
        source_image_pixels, \
            source_image_face_region, \
            source_image_face_emb, \
            source_image_full_mask, \
            source_image_face_mask, \
            source_image_lip_mask = image_processor.preprocess(
            img_pil, save_path, face_expand_ratio,model_path)
    return clip_length,source_image_lip_mask,source_image_face_mask,source_image_full_mask,source_image_face_emb,source_image_face_region,source_image_pixels


class HalloPreImgAndAudio:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        ckpt_list_filter_a = [i for i in folder_paths.get_filename_list("Hallo") if i.endswith(".onnx") and  "audio" in i]
        return {
            "required": {
                "image":("IMAGE",),
                "audio": ("AUDIO",),
                "audio_separator": (["none"] + ckpt_list_filter_a,),
                "face_expand_ratio": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 128,  # Minimum value
                    "max": 2048,  # Maximum value
                    "step": 64,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 128,  # Minimum value
                    "max": 2048,  # Maximum value
                    "step": 64,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "fps": ("INT", {
                    "default": 25,
                    "min": 8,  # Minimum value
                    "max": 100,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "use_cut": ("BOOLEAN", {"default": True},),
            },
        }
    
    RETURN_TYPES = ("EMB_A_HALLO","EMB_I_HALLO",)
    RETURN_NAMES = ("audio_emb","image_emb")
    FUNCTION = "loader_main"
    CATEGORY = "Hallo2"
    
    def loader_main(self,image,audio,audio_separator,face_expand_ratio,width,height,fps,use_cut):
        #pre img
        img_pil=tensor2pil_upscale(image,width, height)
        cache_path = folder_paths.get_output_directory()
        #pre models
        model_path=os.path.join(weigths_face_analysis_path,"face_landmarker_v2_with_blendshapes.task")
        if not os.path.exists(model_path): # download if none
            model_path = hf_hub_download(
                repo_id="fudan-generative-ai/hallo2",
                subfolder="face_analysis/models",
                filename="face_landmarker_v2_with_blendshapes.task",
                local_dir=weigths_current_path,
            )

        clip_length, source_image_lip_mask, source_image_face_mask, source_image_full_mask, source_image_face_emb, source_image_face_region, source_image_pixels\
            =pre_img(img_pil,cache_path,face_expand_ratio,width,height,model_path)
        
        source_image_pixels = source_image_pixels.unsqueeze(0)
        source_image_face_region = source_image_face_region.unsqueeze(0)
        source_image_face_emb = source_image_face_emb.reshape(1, -1)
        source_image_face_emb = torch.tensor(source_image_face_emb)
        
        source_image_full_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_full_mask
        ]
        source_image_face_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_face_mask
        ]
        source_image_lip_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_lip_mask
        ]
        
        image_emb= {"source_image_lip_mask":source_image_lip_mask, "source_image_face_mask":source_image_face_mask, "source_image_full_mask":source_image_full_mask, "source_image_face_emb":source_image_face_emb, "source_image_face_region":source_image_face_region, "source_image_pixels": source_image_pixels}
        # pre audio
        sample_rate = 16000  # config.data.driving_audio.sample_rate
        assert sample_rate == 16000, "audio sample rate must be 16000"
        fps =fps# 25  # config.data.export_video.fps
        
        if audio_separator=="none":
            raise "need chocie a audio_separator model!"
        
        separator_model_file= folder_paths.get_full_path("Hallo", audio_separator)  # config.wav2vec.model_path
        audio_separator_model_file=os.path.basename(separator_model_file) #get name list
        
        wav2vec_only_last_features = False  # config.wav2vec.features == "last"   value=all
        
        #clip_length = 16
        audio_file_prefix = ''.join(random.choice("0123456789abcdefg") for _ in range(6))
        driving_audio_file = os.path.join(folder_paths.get_input_directory(), f"audio_{audio_file_prefix}_temp.wav")
        torchaudio.save(driving_audio_file, audio["waveform"].squeeze(0),  audio["sample_rate"])

        save_path = folder_paths.get_output_directory()
        
        weight_fils_list=["config.json","feature_extractor_config.json","model.safetensors","preprocessor_config.json","special_tokens_map.json","tokenizer_config.json","vocab.json"]
        if not all(filename in os.listdir(weigths_wav2vec_path) for filename in weight_fils_list):
            print(f"no files match in  {weigths_wav2vec_path} ,try download from huggingface!")
            for i in weight_fils_list :
                hf_hub_download(
                    repo_id="fudan-generative-ai/hallo2",
                    subfolder="wav2vec/wav2vec2-base-960h",
                    filename=i,
                    local_dir=weigths_current_path,
                )
        audio_embs, audio_length = load_audio(use_cut, driving_audio_file, save_path, sample_rate, fps,
                                             weigths_wav2vec_path, wav2vec_only_last_features, audio_separator_model_file,
                                             clip_length)
        audio_embs = process_audio_emb(audio_embs)
        
        audio_emb={"audio_emb":audio_embs,"audio_length":audio_length,"clip_length":clip_length,"img_size": (width,height), "driving_audio_path": driving_audio_file,"fps":fps}
        
        torch.cuda.empty_cache()
        
        return (audio_emb,image_emb,)
    

class HalloLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        ckpt_list_filter_=[i for i in folder_paths.get_filename_list("Hallo") if i.endswith(".ckpt")]
        return {
            "required": {
                "checkpoint": (["none"] + folder_paths.get_filename_list("checkpoints"),),
                "vae": (["none"] + folder_paths.get_filename_list("vae"),),
                "motion_module": (["none"] + ckpt_list_filter_,),
                "weight_dtype": (["fp16","bf16", "fp32",],),
                "enable_zero_snr": ("BOOLEAN", {"default": True},),
            },
        }

    RETURN_TYPES = ("MODEL_HALLO",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "Hallo2"

    def loader_main(self,checkpoint,vae, motion_module,weight_dtype,enable_zero_snr):
        config = OmegaConf.load(os.path.join(current_node_path,"configs/inference/long.yaml"))
       
        if weight_dtype == "fp16":
            weight_dtype = torch.float16
        elif weight_dtype == "bf16":
            weight_dtype = torch.bfloat16
        elif weight_dtype == "fp32":
            weight_dtype = torch.float32
        else:
            weight_dtype = torch.float32
        
        # int model
        if motion_module=="none":
            raise "need chocie motion model:mm_sd_v15_v2.ckpt"
        if checkpoint=="none":
            raise "need chocie a checkpoitn base sd1.5"
        if vae=="none":
            raise "need chocie a vae model"
        base_model_path = folder_paths.get_full_path("checkpoints", checkpoint)
        motion_module_path = folder_paths.get_full_path("Hallo", motion_module)
        vae_path=folder_paths.get_full_path("vae", vae)
        val_noise_scheduler, vae, reference_unet, denoising_unet = load_unet(config, enable_zero_snr, base_model_path,
                                                                             motion_module_path,vae_path)
        face_locator = FaceLocator(conditioning_embedding_channels=320)
        image_proj = ImageProjModel(
            cross_attention_dim=denoising_unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=4,
        )
        
        audio_proj = AudioProjModel(
            seq_len=5,
            blocks=12,  # use 12 layers' hidden states of wav2vec
            channels=768,  # audio embedding channel
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32,
        ).to(device=device, dtype=weight_dtype)
        
        audio_ckpt_weights=os.path.join(weigths_hallo2_path, "net.pth")
        if not os.path.exists(audio_ckpt_weights):
            print(f"no net.pth in {weigths_hallo2_path} ,try download from huggingface!")
            hf_hub_download(
                repo_id="fudan-generative-ai/hallo2",
                subfolder="hallo2",
                filename="net.pth",
                local_dir=weigths_current_path,
            )
        
        # Freeze
        vae.requires_grad_(False)
        image_proj.requires_grad_(False)
        reference_unet.requires_grad_(False)
        denoising_unet.requires_grad_(False)
        face_locator.requires_grad_(False)
        audio_proj.requires_grad_(False)
        
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()
        
        net = Net(
            reference_unet,
            denoising_unet,
            face_locator,
            image_proj,
            audio_proj,
        )
        
        m, u = net.load_state_dict(torch.load(audio_ckpt_weights,map_location="cpu"),)
        
        assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
        print(f"loaded weight from : {audio_ckpt_weights} .", )
        
        # 5. inference
        pipeline = FaceAnimatePipeline(
            vae=vae,
            reference_unet=net.reference_unet,
            denoising_unet=net.denoising_unet,
            face_locator=net.face_locator,
            scheduler=val_noise_scheduler,
            image_proj=net.imageproj,
        )
        model={"pipeline":pipeline,"net":net}
        pipeline.to(dtype=weight_dtype)
        torch.cuda.empty_cache()
        return (model,)
    
class HalloSampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_HALLO",),
                "audio_emb": ("EMB_A_HALLO",),
                "image_emb": ("EMB_I_HALLO",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "pose_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                }),
                "face_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                }),
                "lip_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    # The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number",
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 1,  # Minimum value
                    "max": 256,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "use_mask": ("BOOLEAN", {"default": True},),
                "save_video": ("BOOLEAN", {"default": False},),
                
                         },
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT","STRING",)
    RETURN_NAMES = ("image", "frame_rate","path",)
    FUNCTION = "sampler_main"
    CATEGORY = "Hallo2"
    
    def sampler_main(self, model,audio_emb,image_emb,seed,pose_scale,face_scale,lip_scale,steps,cfg,use_mask,save_video):
        
        #pre data
        pipeline=model.get("pipeline")
        pipeline.to(device=device)
        net=model.get("net")
        
        clip_length=audio_emb.get("clip_length")
        img_size=audio_emb.get("img_size")
        audio_emb_in=audio_emb.get("audio_emb")
        audio_length=audio_emb.get("audio_length")
        driving_audio_path=audio_emb.get("driving_audio_path")
        frame_rate=audio_emb.get("fps")
        
        source_image_lip_mask=image_emb.get("source_image_lip_mask")
        source_image_face_mask = image_emb.get("source_image_face_mask")
        source_image_full_mask = image_emb.get("source_image_full_mask")
        source_image_face_emb=image_emb.get("source_image_face_emb")
        source_image_face_region=image_emb.get("source_image_face_region")
        source_image_pixels=image_emb.get("source_image_pixels")
        torch.cuda.empty_cache()
        iamge,path=inference_process(driving_audio_path,pose_scale,face_scale,lip_scale,audio_emb_in,clip_length,audio_length,source_image_pixels,source_image_face_region,
                      source_image_face_emb,source_image_full_mask,source_image_face_mask,source_image_lip_mask,img_size,net,pipeline,steps,cfg,seed,use_mask,save_video,frame_rate)
        
        try:
            del net
        except:
            pass
        pipeline.to("cpu")# move pipeline to cpu ,cause OOM if VRAM<12
        torch.cuda.empty_cache()
        return (iamge,float(frame_rate),path,)


class HallosUpscaleloader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        ckpt_list_filter_U = [i for i in folder_paths.get_filename_list("Hallo") if i.endswith(".pth") and "lib" in i]
        return {
            "required": {
                "realesrgan": (["none"] + folder_paths.get_filename_list("upscale_models"),),
                "face_detection_model": (["none"] + ckpt_list_filter_U,),
                "bg_upsampler": (['realesrgan', 'none', ],),
                "face_upsample": ("BOOLEAN", {"default": False},),
                "has_aligned": ("BOOLEAN", {"default": False},),
                "bg_tile": ("INT", {
                    "default": 400,
                    "min": 200,  # Minimum value
                    "max": 1000,  # Maximum value
                    "step": 10,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "upscale": ("INT", {
                    "default": 2,
                    "min": 2,  # Minimum value
                    "max": 4,  # Maximum value
                    "step": 2,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                         },
        }
    
    RETURN_TYPES = ("HALLO_U_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "Upscale_main"
    CATEGORY = "Hallo2"
    
    def Upscale_main(self,realesrgan, face_detection_model,bg_upsampler,face_upsample,has_aligned,bg_tile,upscale ):
        
        if realesrgan == "none":
            raise "need chocie a 2x upcsale model!"
        model_path = folder_paths.get_full_path("upscale_models", realesrgan)
        
        parse_model = os.path.join(weigths_facelib_path, "parsing_parsenet.pth")
        if not os.path.exists(parse_model):
            print(f"no 'parsing_parsenet.pth' in {parse_model} ,try download from huggingface!")
            hf_hub_download(
                repo_id="fudan-generative-ai/hallo2",
                subfolder="facelib",
                filename="parsing_parsenet.pth",
                local_dir=weigths_current_path,
            )
        
        if face_detection_model == "none":
            raise "need chocie a face_detection_model,resent or yolov5"
        face_detection_model = folder_paths.get_full_path("Hallo", face_detection_model)
        
        hallo_model_path = os.path.join(weigths_hallo2_path, "net_g.pth")
        if not os.path.exists(hallo_model_path):
            print(f"no net_g.pth in {weigths_hallo2_path} ,try download from huggingface!")
            hf_hub_download(
                repo_id="fudan-generative-ai/hallo2",
                subfolder="hallo2",
                filename="net_g.pth",
                local_dir=weigths_current_path,
            )
    
        net,face_upsampler,bg_upsampler,face_helper=pre_u_loader(bg_upsampler, model_path, bg_tile, upscale, face_upsample, device, hallo_model_path,face_detection_model,parse_model,has_aligned)
        model={"net":net,"face_upsampler":face_upsampler,"bg_upsampler":bg_upsampler,"upscale":upscale,"face_helper":face_helper,"has_aligned":has_aligned,"face_upsample":face_upsample}
        return (model,)

class HallosVideoUpscale:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        input_path = folder_paths.get_input_directory()
        video_files = [f for f in os.listdir(input_path) if
                       os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ['webm', 'mp4', 'mkv',
                                                                                            'gif']]
        return {
            "required": {
                "model":("HALLO_U_MODEL",),
                "video_path": (["none"] + video_files,),
                "fidelity_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                }),
                "only_center_face": ("BOOLEAN", {"default": False},),
                "draw_box": ("BOOLEAN", {"default": False},),
                "save_video": ("BOOLEAN", {"default": False},),
                
            },
             "optional": {"image": ("IMAGE",),
                          "audio":("AUDIO",),
                          "frame_rate":("FLOAT",{"forceInput": True,"default": 0.5,}),
                          "path": ("STRING", {"forceInput": True, "default": "", }),
                          },
        }
    
    RETURN_TYPES = ("IMAGE","AUDIO","FLOAT",)
    RETURN_NAMES = ("image","audio","frame_rate")
    FUNCTION = "Upscale_main"
    CATEGORY = "Hallo2"
    
    def Upscale_main(self,model,video_path,fidelity_weight,only_center_face,draw_box, save_video,**kwargs):
        # pre data
        video_img=kwargs.get("image")
        audio=kwargs.get("audio")
        frame_rate=kwargs.get("frame_rate")
        sampler_path=kwargs.get("path")

        #pre model
        net_g = model.get("net")
        face_upsampler= model.get( "face_upsampler")
        bg_upsampler=model.get( "bg_upsampler")
        upscale=model.get( "upscale")
        face_helper=model.get( "face_helper")
        has_aligned=model.get( "has_aligned")
        face_upsample=model.get( "face_upsample")

        front_path=Path(sampler_path) if sampler_path and os.path.exists(Path(sampler_path)) else None
        video_list=[]
        if isinstance(video_img,list) :
            if isinstance(video_img[0],torch.Tensor):
                video_list=video_img
        elif isinstance(video_img, torch.Tensor):
            b, _, _, _ = video_img.size()
            if b == 1:
                img = [b]
                while img is not []:
                    video_list+=img
            else:
                video_list = torch.chunk(video_img, chunks=b)
        
        print(len(video_list))
        video_list=[tensor2cv(i) for i in video_list ] if video_list else [] # tensor to np
        
        if video_path!="none":
            if front_path is not None:
                path = front_path
            else:
                path=os.path.join(folder_paths.get_input_directory(),video_path)
        else:
            if front_path is not None:
                path = front_path
            else:
                path=None

        if video_list: #prior choice
            path = None
        
        if not video_list and video_path=="none" and not front_path:
            raise "Need choice a video or link 'path or image' in the front!!!"
            
        output_path=folder_paths.get_output_directory()
        
        #infer
        print("Start to video upscale processing...")
        video_image,audio_form_v,fps=run_realesrgan(video_list,audio,frame_rate, fidelity_weight, path, output_path,
                       has_aligned, only_center_face, draw_box, bg_upsampler,save_video,net_g, face_upsampler, upscale,face_helper,face_upsample,suffix="",)
        if path is not None:
            audio = audio_form_v
        frame_rate = float(fps)
        
        img_list=[]
        if isinstance(video_image, list):
            for i in video_image:
                for j in i:
                    img_list.append(j)

        image=load_images(img_list)
        return (image,audio,frame_rate,)

NODE_CLASS_MAPPINGS = {
    "HalloPreImgAndAudio":HalloPreImgAndAudio,
    "HalloLoader": HalloLoader,
    "HallosSampler":HalloSampler,
    "HallosUpscaleloader":HallosUpscaleloader,
    "HallosVideoUpscale":HallosVideoUpscale,
    
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HalloPreImgAndAudio":"HalloPreImgAndAudio",
    "HalloLoader": "HalloLoader",
    "HalloSampler":"HalloSampler",
    "HallosUpscaleloader":"HallosUpscaleloader",
    "HallosVideoUpscale":"HallosVideoUpscale"
}
