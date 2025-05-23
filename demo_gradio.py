from diffusers_helper.hf_login import login

import os

# 设置 Hugging Face 下载缓存目录
# script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.' # 处理 Notebook 环境
script_dir = '/kaggle/temp/FramePack/' # 为 Kaggle 环境硬编码路径，如果需要的话
hf_download_path = os.path.abspath(os.path.realpath(os.path.join(script_dir, './hf_download')))
os.environ['HF_HOME'] = hf_download_path
print(f"HF_HOME set to: {hf_download_path}")

import gradio as gr
import torch
import traceback
import einops
# import safetensors.torch as sf
import numpy as np
import argparse
import math
import imageio
import gc

# --- 导入 Accelerate ---
from accelerate import Accelerator # 主要用于 device_map
from accelerate.utils import release_memory # 用于尝试释放内存

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
# 确保导入了所有需要的辅助函数
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
# --- 移除手动内存管理相关的导入 ---
from diffusers_helper.memory import cpu # 保留 cpu 定义
# from diffusers_helper.memory import gpu # 我们将重新定义 gpu
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

print(args)

# --- [修改] 获取计算设备，区分主副 GPU ---
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    gpu0 = torch.device("cuda:0")
    gpu1 = torch.device("cuda:1")
    print(f"Using GPU 0 (main compute): {torch.cuda.get_device_name(gpu0)}")
    print(f"Using GPU 1 (secondary/offload): {torch.cuda.get_device_name(gpu1)}")
    compute_device = gpu0       # Accelerate 管理的模型和主要中间结果放在这里
    secondary_device = gpu1   # 手动加载的模型（VAE, Image Encoder, Transformer）放在这里
    torch.cuda.empty_cache()
    try: torch.cuda.synchronize(gpu0)
    except: pass
    try: torch.cuda.synchronize(gpu1)
    except: pass
elif torch.cuda.is_available():
    gpu0 = torch.device("cuda:0")
    print(f"Using GPU 0: {torch.cuda.get_device_name(gpu0)}")
    compute_device = gpu0
    secondary_device = gpu0 #只有一个 GPU 时，都用它
    torch.cuda.empty_cache()
    try: torch.cuda.synchronize(gpu0)
    except: pass
else:
    gpu0 = torch.device("cpu")
    compute_device = gpu0
    secondary_device = gpu0
# gpu = compute_device # 保留 gpu 变量指向主计算设备? 或者设为 secondary? 暂时不用这个别名
# --- [结束 修改] ---


# --- 使用 Accelerate 加载支持 device_map 的模型，手动加载不支持的 ---
print("Loading models...")
models = {} # 使用字典存储模型
try:
    # Accelerate 会处理这些模型的设备放置 (主要放到 compute_device 或 CPU)
    # 限制 Accelerate 主要使用 GPU 0
    max_memory_accelerate = {0: "14GiB", 1: "1GiB", "cpu": "20GiB"} if torch.cuda.device_count() > 1 else None
    print("Loading text_encoder with device_map='auto'...")
    models['text_encoder'] = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True, max_memory=max_memory_accelerate
    )
    gc.collect(); torch.cuda.empty_cache()
    print("Loading text_encoder_2 with device_map='auto'...")
    models['text_encoder_2'] = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True, max_memory=max_memory_accelerate
    )
    gc.collect(); torch.cuda.empty_cache()

    # 手动加载 Transformer 到 CPU
    print("Loading Transformer to CPU...")
    models['transformer'] = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        'lllyasviel/FramePackI2V_HY', torch_dtype=torch.float16
    ).cpu()
    gc.collect(); torch.cuda.empty_cache()

    # 手动加载 VAE 到 CPU
    print("Loading VAE to CPU...")
    models['vae'] = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16
    ).cpu()
    gc.collect(); torch.cuda.empty_cache()

    # 尝试用 device_map 加载 Image Encoder，失败则回退到 CPU
    print("Attempting to load image_encoder with device_map='auto'...")
    try:
        models['image_encoder'] = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True, max_memory=max_memory_accelerate
        )
        print("Loaded image_encoder with device_map.")
    except ValueError:
        print(f"Warning: image_encoder does not support device_map='auto'. Loading to CPU instead.")
        models['image_encoder'] = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16
        ).cpu()
    except Exception as e_img_enc:
        print(f"Warning: Failed to load image_encoder: {e_img_enc}. Loading to CPU instead.")
        models['image_encoder'] = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16
        ).cpu()
    gc.collect(); torch.cuda.empty_cache()

    # 加载 Tokenizer 和 Feature Extractor (放 CPU)
    print("Loading tokenizers and feature extractor...")
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')

    print("Models and auxiliaries loaded.")

except Exception as e_load:
    print(f"FATAL ERROR DURING MODEL LOADING: {e_load}")
    traceback.print_exc()
    exit()

# --- 设置模型为评估模式 ---
models['vae'].eval()
models['text_encoder'].eval()
models['text_encoder_2'].eval()
models['image_encoder'].eval()
models['transformer'].eval()

# --- Transformer 特定设置 ---
models['transformer'].high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# --- 初始化异步流和输出目录 ---
stream = AsyncStream()
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


# --- Worker 函数定义 ---
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache):

    text_encoder = models['text_encoder']
    text_encoder_2 = models['text_encoder_2']
    vae = models['vae']
    image_encoder = models['image_encoder']
    transformer = models['transformer']

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    job_id = generate_timestamp()

    output_filename_base = os.path.join(outputs_folder, f'{job_id}')
    output_filename = f"{output_filename_base}.mp4"
    video_writer = None

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    # compute_device 和 secondary_device 已在全局定义

    # --- 跟踪模型在哪个 GPU 上 ---
    transformer_on_secondary = False # 在副 GPU
    vae_on_secondary = False       # 在副 GPU
    image_encoder_on_secondary = False # 在副 GPU
    # --- 结束 ---

    try:
        # --- 文本编码 ---
        # Text Encoders 由 Accelerate 管理，主要在 compute_device (gpu0) 或 CPU
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        print("Starting text encoding...")
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        llama_attention_mask = torch.ones_like(llama_vec[..., 0], dtype=torch.long, device=cpu) # Placeholder
        llama_vec_n = None; clip_l_pooler_n = None; llama_attention_mask_n = None
        if cfg == 1:
            print("CFG=1, using zero negative embeddings.")
            llama_vec_n = torch.zeros_like(llama_vec); clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
            llama_attention_mask_n = torch.zeros_like(llama_attention_mask)
        else:
            print("CFG!=1, encoding negative prompt...")
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            llama_attention_mask_n = torch.ones_like(llama_vec_n[..., 0], dtype=torch.long, device=cpu) # Placeholder
        print("Text encoding finished.")


        # --- 处理输入图像 ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1.0
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None] # B, C, T, H, W

        # --- VAE 编码 (使用 secondary_device) ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        print(f"Moving VAE to {secondary_device} for encoding...")
        vae.to(secondary_device); vae_on_secondary = True
        gc.collect(); torch.cuda.empty_cache()
        print(f"VAE moved to {secondary_device}.")
        print("Starting VAE encoding...")
        start_latent = vae_encode(input_image_pt.to(device=vae.device, dtype=vae.dtype), vae)
        print(f"VAE encoding finished...")
        print("Moving VAE back to CPU...")
        vae.to(cpu); vae_on_secondary = False
        gc.collect(); torch.cuda.empty_cache()
        print("VAE moved to CPU.")
        # 将 latent 移到主计算设备 compute_device (gpu0)
        start_latent = start_latent.to(compute_device)
        print(f"Start latent moved to {compute_device}")


        # --- CLIP Vision 编码 (使用 secondary_device) ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        moved_image_encoder_to_secondary = False
        if hasattr(image_encoder, 'hf_device_map'): # 由 Accelerate 管理
             print("Image encoder managed by Accelerate.")
             # 需要确保输入移到 image_encoder 的设备
        elif hasattr(image_encoder, 'device') and image_encoder.device == torch.device('cpu'): # 在 CPU 上
            print(f"Moving Image Encoder to {secondary_device} for encoding...")
            image_encoder.to(secondary_device); image_encoder_on_secondary = True
            moved_image_encoder_to_secondary = True
            gc.collect(); torch.cuda.empty_cache()
            print(f"Image Encoder moved to {secondary_device}.")
        print("Starting CLIP Vision encoding...")
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        print(f"CLIP Vision encoding finished...")
        if moved_image_encoder_to_secondary:
             print("Moving Image Encoder back to CPU...")
             image_encoder.to(cpu); image_encoder_on_secondary = False
             gc.collect(); torch.cuda.empty_cache()
             print("Image Encoder moved to CPU.")
        # 将 hidden_state 移到主计算设备 compute_device (gpu0)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(compute_device)
        print(f"Image encoder hidden state moved to {compute_device}")


        # --- Dtype 转换 (在 compute_device 上) ---
        print("Converting dtypes and moving tensors to compute_device...")
        start_latent = start_latent.to(compute_device, dtype=torch.float16) # 确保在 compute_device
        llama_vec = llama_vec.to(compute_device, dtype=torch.float16)
        llama_vec_n = llama_vec_n.to(compute_device, dtype=torch.float16)
        clip_l_pooler = clip_l_pooler.to(compute_device, dtype=torch.float16)
        clip_l_pooler_n = clip_l_pooler_n.to(compute_device, dtype=torch.float16)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(compute_device, dtype=torch.float16)
        print("Embeddings and poolers processed on compute device.")


        # --- [修改] 将 Transformer 移到 GPU 1 (secondary_device) ---
        print(f"Moving Transformer to {secondary_device} for sampling...")
        transformer.to(secondary_device); transformer_on_secondary = True # 改为 _on_secondary
        gc.collect(); torch.cuda.empty_cache()
        print(f"Transformer moved to {secondary_device}.")
        # --- [结束 修改] ---


        # --- Sampling (采样) ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = torch.Generator("cpu").manual_seed(int(seed))
        num_frames = latent_window_size * 4 - 3
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float16).cpu()
        total_generated_latent_frames = 0
        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        # --- 主采样循环 ---
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                print("User requested end.")
                if video_writer is not None:
                    try: video_writer.close(); print("Video writer closed before early exit.")
                    except Exception as e: print(f"Error closing video writer on early exit: {e}")
                    video_writer = None
                # --- 尝试将 Transformer 移回 CPU ---
                if transformer_on_secondary:
                    try:
                        print("Moving Transformer back to CPU before early exit...")
                        transformer.to(cpu); transformer_on_secondary = False
                        gc.collect(); torch.cuda.empty_cache()
                    except Exception as e_trans_unload:
                        print(f"Error moving Transformer to CPU on early exit: {e_trans_unload}")
                # --- 结束 ---
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            # --- 计算索引和 clean_latents (在 CPU) ---
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            # start_latent 在 compute_device (gpu0)，需要移到 CPU
            clean_latents_pre = start_latent.to(device=history_latents.device, dtype=history_latents.dtype)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # --- 定义回调函数 (VAE 使用 secondary_device) ---
            def callback(d):
                try:
                    preview = d['denoised'] # 在 secondary_device 上
                    preview_vae_device = secondary_device
                    moved_vae_for_preview = False
                    if hasattr(vae, 'device') and vae.device == torch.device('cpu'):
                        vae.to(preview_vae_device); moved_vae_for_preview = True
                    preview = vae_decode_fake(preview.to(preview_vae_device))
                    if moved_vae_for_preview:
                        vae.to(cpu)
                    preview = (preview * 127.5 + 127.5).clamp(0, 255).byte()
                    preview = preview.cpu().numpy()
                    t_idx = preview.shape[2] // 2
                    preview = preview[0, :, t_idx, :, :]
                    preview = preview.transpose(1, 2, 0)

                    if stream.input_queue.top() == 'end':
                        raise KeyboardInterrupt('User ends the task during callback.')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling {current_step}/{steps}'
                    approx_generated_video_frames = max(0, total_generated_latent_frames * 4 - 3)
                    approx_generated_seconds = approx_generated_video_frames / 30.0
                    desc = f'Total generated frames: {int(approx_generated_video_frames)}, Video length: {approx_generated_seconds :.2f} seconds (FPS-30). The video is being extended now ...'
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                except Exception as e_callback:
                    print(f"Error in callback: {e_callback}")
                return

            # --- 执行采样 (在 secondary_device 上) ---
            print(f"Starting sampling on {secondary_device}...")
            # --- 确保所有输入张量移到 secondary_device ---
            current_llama_mask = llama_attention_mask.to(secondary_device) if llama_attention_mask is not None else None
            current_llama_mask_n = llama_attention_mask_n.to(secondary_device) if llama_attention_mask_n is not None else None
            if current_llama_mask is None:
                 print("Warning: Positive Llama attention mask is None, creating default.")
                 current_llama_mask = torch.ones_like(llama_vec[..., 0], dtype=torch.long, device=secondary_device)
            if current_llama_mask_n is None and cfg != 1:
                 print("Warning: Negative Llama attention mask is None, creating default.")
                 current_llama_mask_n = torch.ones_like(llama_vec_n[..., 0], dtype=torch.long, device=secondary_device)
            elif cfg == 1 and current_llama_mask_n is None:
                 current_llama_mask_n = torch.zeros_like(current_llama_mask).to(secondary_device)

            generated_latents = sample_hunyuan(
                transformer=transformer, # 已经在 secondary_device
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec.to(secondary_device), # 移到 GPU 1
                prompt_embeds_mask=current_llama_mask,
                prompt_poolers=clip_l_pooler.to(secondary_device), # 移到 GPU 1
                negative_prompt_embeds=llama_vec_n.to(secondary_device), # 移到 GPU 1
                negative_prompt_embeds_mask=current_llama_mask_n,
                negative_prompt_poolers=clip_l_pooler_n.to(secondary_device), # 移到 GPU 1
                device=secondary_device, # 指定在 GPU 1 上计算
                dtype=torch.float16,
                image_embeddings=image_encoder_last_hidden_state.to(secondary_device), # 移到 GPU 1
                latent_indices=latent_indices,
                clean_latents=clean_latents.to(device=secondary_device, dtype=torch.float16), # 移到 GPU 1
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x.to(device=secondary_device, dtype=torch.float16), # 移到 GPU 1
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x.to(device=secondary_device, dtype=torch.float16), # 移到 GPU 1
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            print(f"Sampling finished for this section on {secondary_device}.")
            # generated_latents 在 secondary_device (gpu1) 上

            # --- 处理采样结果 ---
            if is_last_section:
                # start_latent 在 compute_device (gpu0)，需要移动到 secondary_device
                generated_latents = torch.cat([start_latent.to(device=generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)

            added_latent_frames_count = int(generated_latents.shape[2])
            total_generated_latent_frames += added_latent_frames_count

            # --- 更新 history_latents (在 CPU) ---
            # 将新生成的 latent (在 secondary_device) 移回 CPU 并拼接
            history_latents = torch.cat([generated_latents.to(device=history_latents.device, dtype=history_latents.dtype), history_latents], dim=2)

            # --- VAE 解码和流式写入 (使用 secondary_device) ---
            if added_latent_frames_count > 0:
                print(f"Moving VAE to {secondary_device} for decoding...")
                vae.to(secondary_device); vae_on_gpu1 = True
                gc.collect(); torch.cuda.empty_cache()
                print(f"VAE moved to {secondary_device}.")

                latents_to_decode_this_iter = history_latents[:, :, :added_latent_frames_count, :, :] # 在 CPU
                current_pixels_segment = vae_decode(latents_to_decode_this_iter.to(device=vae.device, dtype=vae.dtype), vae).float().cpu() # 解码到 float32 CPU
                print("VAE decoding for writing finished.")

                print("Moving VAE back to CPU...")
                vae.to(cpu); vae_on_gpu1 = False
                gc.collect(); torch.cuda.empty_cache()
                print("VAE moved to CPU.")

                # --- 转换为 NumPy uint8 格式 [T, H, W, C] for imageio ---
                pixels_np = current_pixels_segment.squeeze(0)
                pixels_np = pixels_np.permute(1, 2, 3, 0)
                pixels_np = (pixels_np * 127.5 + 127.5).clamp(0, 255).byte()
                pixels_np = pixels_np.numpy()

                # --- 初始化视频写入器 (如果需要) ---
                if video_writer is None:
                    fps_to_save = 30
                    video_writer = imageio.get_writer(output_filename, fps=fps_to_save, codec='libx264', quality=8, output_params=['-preset', 'fast', '-tune', 'fastdecode'], macro_block_size=16, ffmpeg_log_level='warning')
                    print(f"Video writer initialized for {output_filename}")

                # --- 将当前段的帧写入视频文件 ---
                print(f"Appending {pixels_np.shape[0]} frames to video...")
                for i in range(pixels_np.shape[0]):
                    video_writer.append_data(pixels_np[i])
                print(f"Frames appended.")

                # --- 关键：释放内存 ---
                del current_pixels_segment
                del pixels_np
                del latents_to_decode_this_iter
                gc.collect()
                print("Memory released after decoding and writing.")
                # --- 结束关键步骤 ---

            # --- 更新 Gradio 界面 ---
            print(f'Decoded and wrote segment. Total latent frames generated: {total_generated_latent_frames}')
            if video_writer is not None:
                stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break # 结束循环

        # --- 循环正常结束后，关闭写入器 (移动到 finally) ---

    except KeyboardInterrupt:
        print("User interrupted the process.")
    except Exception:
        traceback.print_exc()

    finally: # 最终清理
        print("Entering finally block...")
        # --- 关闭 Writer ---
        if video_writer is not None:
            try:
                print("Closing video writer in finally block...")
                video_writer.close()
                print("Video writer closed successfully in finally.")
            except Exception as e_close:
                print(f"Error closing video writer in finally: {e_close}")
            video_writer = None

        # --- 将模型移回 CPU ---
        if transformer_on_secondary: # 改为检查 _on_secondary
            try:
                print("Moving Transformer back to CPU in finally...")
                transformer.to(cpu)
                transformer_on_secondary = False # 改为 _on_secondary
            except Exception as e_trans_unload:
                print(f"Error moving Transformer to CPU in finally: {e_trans_unload}")
        if vae_on_secondary: # 改为检查 _on_secondary
             try:
                 print("Moving VAE back to CPU in finally...")
                 vae.to(cpu)
                 vae_on_secondary = False # 改为 _on_secondary
             except Exception as e_vae_unload:
                  print(f"Error moving VAE to CPU in finally: {e_vae_unload}")
        if image_encoder_on_secondary: # 改为检查 _on_secondary
             try:
                 print("Moving Image Encoder back to CPU in finally...")
                 image_encoder.to(cpu)
                 image_encoder_on_secondary = False # 改为 _on_secondary
             except Exception as e_img_unload:
                  print(f"Error moving Image Encoder to CPU in finally: {e_img_unload}")

        # --- 尝试释放 Accelerate 管理的模型 ---
        try:
            print("Releasing models potentially managed by Accelerate...")
            release_memory(models.get('text_encoder'), models.get('text_encoder_2'))
            print("Attempting final garbage collection and cache clearing...")
            gc.collect()
            torch.cuda.empty_cache()
            print("Memory release attempted.")
        except Exception as e_release:
            print(f"Error during final memory release: {e_release}")

        # --- 发送结束信号 ---
        print("Pushing end signal to stream.")
        stream.output_queue.push(('end', None))
        print("Worker function finished.")


# --- Gradio 界面处理函数 ---
def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache):
    global stream
    assert input_image is not None, 'No input image!'
    print("Starting generation process...")

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    stream = AsyncStream()

    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache)

    output_filename = None
    while True:
        flag, data = stream.output_queue.next()
        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
        elif flag == 'progress':
            preview, desc, html = data
            yield gr.update(value=output_filename), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
        elif flag == 'end':
            print("Generation process ended.")
            yield output_filename, gr.update(visible=False), gr.update(value=''), '', gr.update(interactive=True), gr.update(interactive=False)
            break

# --- 结束处理函数 ---
def end_process():
    print("End generation requested by user.")
    stream.input_queue.push('end')

# --- Gradio 界面定义 ---
quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]

css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                seed = gr.Number(label="Seed", value=31337, precision=0)
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                # --- 移除 gpu_memory_preservation 滑块 ---

        with gr.Column():
            preview_image = gr.Image(label="Sampling Preview (Middle Frame)", height=200, visible=False)
            result_video = gr.Video(label="Generated Video", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    # --- 更新 Gradio 输入列表 ---
    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)

# --- 启动 Gradio 应用 ---
block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
