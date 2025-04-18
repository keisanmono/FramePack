from diffusers_helper.hf_login import login

import os

# 设置 Hugging Face 下载缓存目录
script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.' # 处理 Notebook 环境
hf_download_path = os.path.abspath(os.path.realpath(os.path.join(script_dir, './hf_download')))
os.environ['HF_HOME'] = hf_download_path
print(f"HF_HOME set to: {hf_download_path}")

import gradio as gr
import torch
import traceback
import einops
# import safetensors.torch as sf # Safetensors 可能由 accelerate 处理，暂时注释掉
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
# from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.memory import cpu, gpu # 可能仍然需要 cpu, gpu 定义
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

# --- 获取计算设备 ---
# Accelerate 通常会自动选择，但我们保留 gpu 变量用于手动移动 VAE 等
if torch.cuda.is_available():
    gpu = torch.device("cuda:0") # 默认使用第一个 GPU
    torch.cuda.empty_cache() # 清理显存
else:
    gpu = torch.device("cpu")
print(f"Primary compute device selected: {gpu}")


# --- 使用 Accelerate 加载支持 device_map 的模型，手动加载不支持的 ---
print("Loading models...")
try:
    # Accelerate 会处理这些模型的设备放置
    print("Loading text_encoder with device_map='auto'...")
    text_encoder = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True
    )
    gc.collect(); torch.cuda.empty_cache()
    print("Loading text_encoder_2 with device_map='auto'...")
    text_encoder_2 = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True
    )
    gc.collect(); torch.cuda.empty_cache()
    print("Loading transformer with device_map='auto'...")
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        'lllyasviel/FramePackI2V_HY', torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True
    )
    gc.collect(); torch.cuda.empty_cache()

    # 手动加载不支持 device_map 的 VAE 到 CPU
    print("Loading VAE to CPU (device_map not supported)...")
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16
    ).cpu()
    gc.collect(); torch.cuda.empty_cache()

    # 尝试用 device_map 加载 Image Encoder，失败则回退到 CPU
    print("Attempting to load image_encoder with device_map='auto'...")
    try:
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True
        )
        print("Loaded image_encoder with device_map.")
    except ValueError:
        print(f"Warning: image_encoder does not support device_map='auto'. Loading to CPU instead.")
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16
        ).cpu()
    except Exception as e_img_enc:
        print(f"Warning: Failed to load image_encoder with device_map='auto': {e_img_enc}. Loading to CPU instead.")
        image_encoder = SiglipVisionModel.from_pretrained(
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
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

# --- Transformer 特定设置 ---
transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# --- 初始化异步流和输出目录 ---
stream = AsyncStream()
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


# --- Worker 函数定义 ---
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache): # 移除 gpu_memory_preservation

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    job_id = generate_timestamp()

    # --- 初始化视频写入相关变量 ---
    output_filename_base = os.path.join(outputs_folder, f'{job_id}')
    output_filename = f"{output_filename_base}.mp4"
    video_writer = None
    # --- 结束 初始化 ---

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    # --- 获取主计算设备 ---
    compute_device = gpu # 使用之前定义的 gpu 设备 ('cuda:0' 或 'cpu')

    try:
        # --- 文本编码 ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        # 调用编码函数 (hunyuan.py 中的代码会将 input_ids 移到 model.device)
        print("Starting text encoding...")
        # 确保 encode_prompt_conds 能处理分布式的 text_encoder 和 text_encoder_2
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        print("Text encoding finished.")

        # --- 处理输入图像 ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1.0
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None] # B, C, T, H, W

        # --- VAE 编码 ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        # --- 手动加载 VAE 到 GPU ---
        print("Moving VAE to GPU for encoding...")
        vae.to(gpu)
        gc.collect(); torch.cuda.empty_cache()
        print("VAE moved to GPU.")
        # --- 结束 ---
        print("Starting VAE encoding...")
        # 确保 input_image_pt 在 VAE 设备上
        start_latent = vae_encode(input_image_pt.to(vae.device, dtype=vae.dtype), vae)
        print(f"VAE encoding finished. Start latent shape: {start_latent.shape}, dtype: {start_latent.dtype}, device: {start_latent.device}")
        # --- VAE 编码后移回 CPU ---
        print("Moving VAE back to CPU...")
        vae.to(cpu)
        gc.collect(); torch.cuda.empty_cache()
        print("VAE moved to CPU.")
        # --- 结束 ---
        # 将 latent 移到主计算设备 compute_device
        start_latent = start_latent.to(compute_device)
        print(f"Start latent moved to {compute_device}")


        # --- CLIP Vision 编码 ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        # --- 如果 image_encoder 在 CPU，手动移到 GPU ---
        moved_image_encoder = False
        if hasattr(image_encoder, 'device') and image_encoder.device == torch.device('cpu'): # 检查是否需要移动
            print("Moving Image Encoder to GPU for encoding...")
            image_encoder.to(gpu)
            moved_image_encoder = True
            gc.collect(); torch.cuda.empty_cache()
            print("Image Encoder moved to GPU.")
        # --- 结束 ---
        print("Starting CLIP Vision encoding...")
        # 确保 hf_clip_vision_encode 能处理 NumPy 输入和可能在 GPU 的模型
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        print(f"CLIP Vision encoding finished. Hidden state device: {image_encoder_last_hidden_state.device}")
        # --- 编码后移回 CPU (如果之前移动了) ---
        if moved_image_encoder:
             print("Moving Image Encoder back to CPU...")
             image_encoder.to(cpu)
             gc.collect(); torch.cuda.empty_cache()
             print("Image Encoder moved to CPU.")
        # --- 结束 ---
        # 将 hidden_state 移到主计算设备
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(compute_device)
        print(f"Image encoder hidden state moved to {compute_device}")


        # --- Dtype 转换 ---
        start_latent = start_latent.to(torch.float16)
        llama_vec = llama_vec.to(compute_device, dtype=torch.float16)
        llama_vec_n = llama_vec_n.to(compute_device, dtype=torch.float16)
        clip_l_pooler = clip_l_pooler.to(compute_device, dtype=torch.float16)
        clip_l_pooler_n = clip_l_pooler_n.to(compute_device, dtype=torch.float16)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(compute_device, dtype=torch.float16)
        print("Embeddings and poolers moved and converted to float16 on compute device.")


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
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            # --- 计算索引和 clean_latents ---
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            clean_latents_pre = start_latent.to(device=history_latents.device, dtype=history_latents.dtype)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2) # clean_latents 在 CPU

            # --- 定义回调函数 ---
            def callback(d):
                try:
                    preview = d['denoised']
                    # VAE 可能在 CPU，需要移到 GPU
                    preview_vae_device = gpu # 假设 VAE 会被移到 gpu
                    preview = vae_decode_fake(preview.to(preview_vae_device)) # 移到 GPU 再 fake decode
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

            # --- 执行采样 ---
            # 确保传递给 sample_hunyuan 的张量在正确的设备上
            print("Starting sampling...")
            # 注意：transformer 可能部分在 CPU 部分在 GPU (被 Accelerate 分配)
            # sample_hunyuan 内部需要能处理这种情况，或者我们把所有输入都放到 compute_device
            generated_latents = sample_hunyuan(
                transformer=transformer, # 由 Accelerate 管理
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec.to(compute_device), # 确保在 compute_device
                prompt_embeds_mask=llama_attention_mask.to(compute_device),
                prompt_poolers=clip_l_pooler.to(compute_device),
                negative_prompt_embeds=llama_vec_n.to(compute_device),
                negative_prompt_embeds_mask=llama_attention_mask_n.to(compute_device),
                negative_prompt_poolers=clip_l_pooler_n.to(compute_device),
                device=compute_device, # 指定主计算设备
                dtype=torch.float16,
                image_embeddings=image_encoder_last_hidden_state.to(compute_device), # 确保在 compute_device
                latent_indices=latent_indices,
                clean_latents=clean_latents.to(device=compute_device, dtype=torch.float16), # 移到 compute_device
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x.to(device=compute_device, dtype=torch.float16), # 移到 compute_device
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x.to(device=compute_device, dtype=torch.float16), # 移到 compute_device
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            print("Sampling finished for this section.")
            # generated_latents 在 compute_device 上

            # --- 处理采样结果 ---
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(device=generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)

            added_latent_frames_count = int(generated_latents.shape[2])
            total_generated_latent_frames += added_latent_frames_count

            # --- 更新 history_latents ---
            history_latents = torch.cat([generated_latents.to(device=history_latents.device, dtype=history_latents.dtype), history_latents], dim=2)

            # --- VAE 解码和流式写入 ---
            if added_latent_frames_count > 0:
                # --- 手动加载 VAE 到 GPU ---
                print("Moving VAE to GPU for decoding...")
                vae.to(gpu)
                gc.collect(); torch.cuda.empty_cache()
                print("VAE moved to GPU.")
                # --- 结束 ---

                latents_to_decode_this_iter = history_latents[:, :, :added_latent_frames_count, :, :] # 从 history 切片 (在 CPU)
                # --- 确保传递给 vae_decode 的 latent 在 VAE 设备上且类型正确 ---
                current_pixels_segment = vae_decode(latents_to_decode_this_iter.to(device=vae.device, dtype=vae.dtype), vae).float().cpu() # 解码到 float32 CPU
                print("VAE decoding for writing finished.")

                # --- 解码后移回 CPU ---
                print("Moving VAE back to CPU...")
                vae.to(cpu)
                gc.collect(); torch.cuda.empty_cache()
                print("VAE moved to CPU.")
                # --- 结束 ---

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
                break

        # --- 循环正常结束后，关闭写入器 ---
        if video_writer is not None:
            try:
                print("Closing video writer after loop completion...")
                video_writer.close()
                print("Video writer closed successfully after loop completion.")
                video_writer = None
            except Exception as e_close:
                print(f"Error closing video writer after loop completion: {e_close}")

    except KeyboardInterrupt: # 处理用户中断
        print("User interrupted the process.")
        if video_writer is not None:
            try: video_writer.close(); print("Video writer closed after interruption.")
            except Exception as e: print(f"Error closing video writer after interruption: {e}")

    except Exception: # 处理其他异常
        traceback.print_exc()
        if video_writer is not None:
            try: video_writer.close(); print("Video writer closed after exception.")
            except Exception as e: print(f"Error closing video writer after exception: {e}")

    finally: # 最终清理
        print("Entering finally block...")
        # --- 尝试释放 Accelerate 管理的模型 ---
        try:
            print("Releasing models managed by Accelerate...")
            release_memory(text_encoder, text_encoder_2, transformer) # 释放支持 device_map 的模型
            # 手动加载的模型（VAE, Image Encoder）也尝试 del
            if 'vae' in locals(): del vae
            if 'image_encoder' in locals(): del image_encoder
            gc.collect()
            torch.cuda.empty_cache()
            print("Memory released.")
        except Exception as e_release:
            print(f"Error during final memory release: {e_release}")
        # --- 发送结束信号 ---
        print("Pushing end signal to stream.")
        stream.output_queue.push(('end', None))
        print("Worker function finished.")


# --- Gradio 界面处理函数 ---
def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache): # 移除 gpu_memory_preservation
    global stream
    assert input_image is not None, 'No input image!'
    print("Starting generation process...")

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    stream = AsyncStream()

    # --- 移除 gpu_memory_preservation 参数 ---
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
