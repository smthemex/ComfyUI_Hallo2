# pylint: disable=E1101
# scripts/inference.py

"""
This script contains the main inference pipeline for processing audio and image inputs to generate a video output.

The script imports necessary packages and classes, defines a neural network model, 
and contains functions for processing audio embeddings and performing inference.

The main inference process is outlined in the following steps:
1. Initialize the configuration.
2. Set up runtime variables.
3. Prepare the input data for inference (source image, face mask, and face embeddings).
4. Process the audio embeddings.
5. Build and freeze the model and scheduler.
6. Run the inference loop and save the result.

Usage:
This script can be run from the command line with the following arguments:
- audio_path: Path to the audio file.
- image_path: Path to the source image.
- face_mask_path: Path to the face mask image.
- face_emb_path: Path to the face embeddings file.
- output_path: Path to save the output video.

Example:
python scripts/inference.py --audio_path audio.wav --image_path image.jpg 
    --face_mask_path face_mask.png --face_emb_path face_emb.pt --output_path output.mp4
"""

import argparse
import os
import sys

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pydub import AudioSegment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..hallo.animate.face_animate import FaceAnimatePipeline
from ..hallo.datasets.audio_processor import AudioProcessor
from ..hallo.datasets.image_processor import ImageProcessor
from ..hallo.models.audio_proj import AudioProjModel
from ..hallo.models.face_locator import FaceLocator
from ..hallo.models.image_proj import ImageProjModel
from ..hallo.models.unet_2d_condition import UNet2DConditionModel
from ..hallo.models.unet_3d import UNet3DConditionModel
from ..hallo.utils.config import filter_non_none
from ..hallo.utils.util import tensor_to_video_batch, merge_videos
import folder_paths

from icecream import ic

class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    
    Args:
        reference_unet (UNet2DConditionModel): The UNet2DConditionModel used as a reference for inference.
        denoising_unet (UNet3DConditionModel): The UNet3DConditionModel used for denoising the input audio.
        face_locator (FaceLocator): The FaceLocator model used to locate the face in the input image.
        imageproj (nn.Module): The ImageProjector model used to project the source image onto the face.
        audioproj (nn.Module): The AudioProjector model used to project the audio embeddings onto the face.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        imageproj,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(self,):
        """
        empty function to override abstract function of nn Module
        """

    def get_modules(self):
        """
        Simple method to avoid too-few-public-methods pylint error
        """
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "imageproj": self.imageproj,
            "audioproj": self.audioproj,
        }


def process_audio_emb(audio_emb):
    """
    Process the audio embedding to concatenate with other tensors.

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb

def save_image_batch(image_tensor, save_path):
    image_tensor = (image_tensor + 1) / 2

    os.makedirs(save_path, exist_ok=True)

    for i in range(image_tensor.shape[0]):
        img_tensor = image_tensor[i]
        
        img_array = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        img_array = (img_array * 255).astype(np.uint8)
        
        image = Image.fromarray(img_array)
        image.save(os.path.join(save_path, f'motion_frame_{i}.png'))


def cut_audio(audio_path, save_dir, length=60):
    audio = AudioSegment.from_wav(audio_path)

    segment_length = length * 1000 # pydub使用毫秒

    num_segments = len(audio) // segment_length + (1 if len(audio) % segment_length != 0 else 0)

    os.makedirs(save_dir, exist_ok=True)

    audio_list = [] 

    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, len(audio))
        segment = audio[start_time:end_time]
        
        path = f"{save_dir}/segment_{i+1}.wav"
        audio_list.append(path)
        segment.export(path, format="wav")

    return audio_list


def inference_process(driving_audio_path,pose_weight,face_weight,lip_weight,audio_emb,clip_length,audio_length,source_image_pixels,source_image_face_region,
                      source_image_face_emb,source_image_full_mask,source_image_face_mask,source_image_lip_mask,img_size,net,pipeline,inference_steps,cfg_scale,seed,use_mask,save_video,fps):

   
    motion_scale = [pose_weight, face_weight, lip_weight] # 1.0,1.0,1.0
    # from here
    times = audio_emb.shape[0] // clip_length
    tensor_result = []
    generator = torch.manual_seed(seed)

    ic(audio_emb.shape)
    ic(audio_length)    
    batch_size = 60
    start = 0
    for t in range(times):
        print(f"Start infer batch {t+1} ,and total batchs is {times}.")

        if len(tensor_result) == 0:
            # The first iteration
            motion_zeros = source_image_pixels.repeat(
                2, 1, 1, 1)
            motion_zeros = motion_zeros.to(
                dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat(
                [source_image_pixels, motion_zeros], dim=0)  # concat the ref image and the first motion frames
        else:
            motion_frames = tensor_result[-1][0]
            motion_frames = motion_frames.permute(1, 0, 2, 3)
            motion_frames = motion_frames[0-2:]
            motion_frames = motion_frames * 2.0 - 1.0
            motion_frames = motion_frames.to(
                dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat(
                [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames
        
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

        pixel_motion_values = pixel_values_ref_img[:, 1:]

        if use_mask:
            b, f, c, h, w = pixel_motion_values.shape
            rand_mask = torch.rand(h, w)
            mask = rand_mask > 0.25
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
            mask = mask.expand(b, f, c, h, w)  

            face_mask = source_image_face_region.repeat(f, 1, 1, 1).unsqueeze(0)
            assert face_mask.shape == mask.shape
            mask = mask | face_mask.bool()

            pixel_motion_values = pixel_motion_values * mask
            pixel_values_ref_img[:, 1:] = pixel_motion_values

        
        assert pixel_motion_values.shape[0] == 1

        audio_tensor = audio_emb[
            t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])
        ]
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(
            device=net.audioproj.device, dtype=net.audioproj.dtype)
        audio_tensor = net.audioproj(audio_tensor)
       
        pipeline_output = pipeline(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            face_emb=source_image_face_emb,
            face_mask=source_image_face_region,
            pixel_values_full_mask=source_image_full_mask,
            pixel_values_face_mask=source_image_face_mask,
            pixel_values_lip_mask=source_image_lip_mask,
            width=img_size[0],
            height=img_size[1],
            video_length=clip_length,
            num_inference_steps=inference_steps,
            guidance_scale=cfg_scale,
            generator=generator,
            motion_scale=motion_scale,
        )

        ic(pipeline_output.videos.shape)# [1,3, 16, 512, 512]
        tensor_result.append(pipeline_output.videos)

        if (t+1) % batch_size == 0 or (t+1)==times:
            last_motion_frame = [tensor_result[-1]]
            ic(len(tensor_result))

            if start!=0:
                tensor_result = torch.cat(tensor_result[1:], dim=2)
            else:
                tensor_result = torch.cat(tensor_result, dim=2)
            
            tensor_result = tensor_result.squeeze(0)
            f = tensor_result.shape[1]
            length = min(f, audio_length)
            tensor_result = tensor_result[:, :length] #torch.Size([3, 188, 512, 512])
            tensor_list= tensor_result.clone().permute(1, 2, 3, 0) # convert to [f, h, w, c]  comfyUI [B,H,W,C], C=3
            
            ic(tensor_result.shape)
            ic(start)
            ic(audio_length)
            
            output_file = os.path.join(folder_paths.get_output_directory(), f"Hallo2-{t+1:06}.mp4")
            if save_video:
                tensor_to_video_batch(tensor_result, output_file, start, driving_audio_path, fps)
                output_path = output_file
                del tensor_result
                tensor_result = last_motion_frame
                audio_length -= length
                start += length
            else:
                output_path=""
                
    return tensor_list,output_path
    

