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
import safetensors.torch as sf
import numpy as np
import argparse
import math
import imageio
import gc

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
# 确保导入了所有需要的辅助函数
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
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

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60 # 根据可用显存判断是否为高显存模式

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# --- 模型和分词器加载 ---
print("Loading models to CPU...")
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.float16).cpu()
print("Models loaded to CPU.")

# --- 设置模型为评估模式和数据类型 ---
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()
    print("VAE slicing and tiling enabled for low VRAM.")

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# --- 确保模型使用 float16 ---
transformer.to(dtype=torch.float16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)
print("Models converted to float16.")

# --- 禁用梯度计算 ---
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

# --- 根据显存情况处理模型位置 ---
if not high_vram:
    print("Installing DynamicSwap for low VRAM mode...")
    # DynamicSwapInstaller 类似于 huggingface 的 enable_sequential_offload，但更快
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu) # 安装在 Llama 上
    # 注意：DynamicSwapInstaller 不会安装在 text_encoder_2, vae, image_encoder 上
    print("DynamicSwap installed.")
else:
    print("Moving models to GPU for high VRAM mode...")
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)
    print("Models moved to GPU.")

# --- 初始化异步流和输出目录 ---
stream = AsyncStream()
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


# --- Worker 函数定义 ---
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    # --- 初始化视频写入相关变量 ---
    output_filename_base = os.path.join(outputs_folder, f'{job_id}')
    output_filename = f"{output_filename_base}.mp4"  # 最终视频文件名
    video_writer = None  # 初始化为 None
    # --- 结束 初始化 ---

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # --- 清理 GPU ---
        if not high_vram:
            print("Unloading models from GPU...")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # --- 文本编码 ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        if not high_vram:
            # --- [修改] 只加载 text_encoder_2，并手动移动 text_encoder 的 embedding 层 ---
            print("Low VRAM: Loading text_encoder_2 to GPU...")
            load_model_as_complete(text_encoder_2, target_device=gpu) # <--- 加载 CLIP
            print("Low VRAM: Moving text_encoder's embedding layer to GPU...")
            try:
                text_encoder.embed_tokens.to(gpu) # <--- 只移动 Embedding 层到 GPU
                gc.collect() # 清理一下内存
                print("Text encoder embedding moved to GPU.")
            except Exception as e_embed:
                 print(f"Warning: Failed to move text_encoder embedding to GPU: {e_embed}") # 加个警告以防万一
            # --- [结束 修改] ---

        # 调用编码函数 (hunyuan.py 中的代码会将 input_ids 移到 model.device)
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        # --- [修改] 编码后，将 Embedding 层移回 CPU 并卸载 text_encoder_2 ---
        if not high_vram:
            print("Low VRAM: Moving text_encoder's embedding layer back to CPU...")
            try:
                text_encoder.embed_tokens.to(cpu) # <--- 移回 CPU，释放 VRAM
                gc.collect()
                print("Low VRAM: Unloading text_encoder_2 from GPU...")
                unload_complete_models(text_encoder_2) # <--- 卸载 CLIP
            except Exception as e_unload:
                print(f"Warning: Error during post-encoding cleanup: {e_unload}")
        # --- [结束 修改] ---

        # --- 处理输入图像 ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640) # 根据输入图像尺寸确定合适的处理尺寸
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png')) # 保存处理后的输入图像
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1.0 # 归一化到 [-1, 1]
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None] # 增加 Batch 和 Time 维度 (B, C, T, H, W)

        # --- VAE 编码 ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu) # 需要时加载 VAE
        start_latent = vae_encode(input_image_pt, vae) # 将输入图像编码为 latent

        # --- CLIP Vision 编码 ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        if not high_vram:
            unload_complete_models(vae) # 卸载 VAE，为 Image Encoder 腾空间
            load_model_as_complete(image_encoder, target_device=gpu) # 需要时加载 Image Encoder
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state # 获取图像 embedding

        # --- Dtype 转换 (确保所有 embedding 和 pooler 与 transformer 类型一致) ---
        start_latent = start_latent.to(transformer.dtype) # 确保 start_latent 类型也匹配
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # --- Sampling (采样) ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = torch.Generator("cpu").manual_seed(int(seed)) # 确保 seed 是整数
        num_frames = latent_window_size * 4 - 3 # 计算实际生成的帧数
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float16).cpu() # 初始化 history_latents (float16, cpu)
        total_generated_latent_frames = 0 # 跟踪已生成的 latent 帧总数
        latent_paddings = reversed(range(total_latent_sections)) # 计算反向采样的 padding 列表

        if total_latent_sections > 4: # 特殊处理 padding 顺序以获得可能更好的效果
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        # --- 主采样循环 ---
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end': # 检查是否用户请求停止
                print("User requested end.")
                if video_writer is not None:
                    try: video_writer.close(); print("Video writer closed before early exit.")
                    except Exception as e: print(f"Error closing video writer on early exit: {e}")
                    video_writer = None
                stream.output_queue.push(('end', None))
                return # 退出 worker

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            # --- 计算索引和 clean_latents ---
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            clean_latents_pre = start_latent.to(device=history_latents.device, dtype=history_latents.dtype) # 将 start_latent 转到 history (cpu, float16)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # --- 移动 Transformer 模型 ---
            if not high_vram:
                unload_complete_models(image_encoder) # 卸载 Image Encoder，为 Transformer 腾空间
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            # --- TeaCache 初始化 ---
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            # --- 定义回调函数 ---
            def callback(d):
                try:
                    preview = d['denoised'] # 获取去噪后的 latent
                    preview = vae_decode_fake(preview) # 使用快速 fake 解码获取预览
                    preview = (preview * 127.5 + 127.5).clamp(0, 255).byte() # 转换到 0-255 uint8
                    preview = preview.cpu().numpy() # 转 NumPy
                    t_idx = preview.shape[2] // 2 # 取时间中间帧索引
                    preview = preview[0, :, t_idx, :, :] # 选择 Batch 0, 所有通道, 中间帧, 所有高宽 [C, H, W]
                    preview = preview.transpose(1, 2, 0) # 转换 HWC 格式给 Gradio Image

                    if stream.input_queue.top() == 'end': # 检查用户是否停止
                        raise KeyboardInterrupt('User ends the task during callback.')

                    current_step = d['i'] + 1 # 当前步数
                    percentage = int(100.0 * current_step / steps) # 计算百分比
                    hint = f'Sampling {current_step}/{steps}' # 进度提示
                    # 计算大致的已生成视频长度 (latent 帧数 * 4 - 3) / 30 fps
                    approx_generated_video_frames = max(0, total_generated_latent_frames * 4 - 3) # 估算已生成的像素帧数
                    approx_generated_seconds = approx_generated_video_frames / 30.0 # 估算已生成的秒数
                    desc = f'Total generated frames: {int(approx_generated_video_frames)}, Video length: {approx_generated_seconds :.2f} seconds (FPS-30). The video is being extended now ...'
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint)))) # 推送进度
                except Exception as e_callback:
                    print(f"Error in callback: {e_callback}") # 打印回调错误
                return

            # --- 执行采样 ---
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu, # 在 GPU 上运行采样
                dtype=torch.float16, # 使用 float16
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents.to(device=gpu, dtype=torch.float16), # clean_latents 需要在 GPU 上且类型匹配
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x.to(device=gpu, dtype=torch.float16), # 同上
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x.to(device=gpu, dtype=torch.float16), # 同上
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            # --- 处理采样结果 ---
            if is_last_section: # 如果是最后一段，将起始 latent 拼接到开头
                generated_latents = torch.cat([start_latent.to(device=generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)

            added_latent_frames_count = int(generated_latents.shape[2]) # 获取本次生成的 latent 帧数
            total_generated_latent_frames += added_latent_frames_count # 更新总帧数

            # --- 更新 history_latents ---
            # 将新生成的 latent (在 GPU, float16) 移到 history_latents (在 CPU, float16) 并拼接
            history_latents = torch.cat([generated_latents.to(device=history_latents.device, dtype=history_latents.dtype), history_latents], dim=2)

            # --- VAE 解码和流式写入 ---
            if added_latent_frames_count > 0:
                if not high_vram:
                    offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8) # 卸载 Transformer
                    load_model_as_complete(vae, target_device=gpu) # 加载 VAE

                # --- 只解码本次循环新生成的 latent 部分 ---
                # 从 history_latents 切片出当前迭代需要解码的部分
                latents_to_decode_this_iter = history_latents[:, :, :added_latent_frames_count, :, :].cpu()

                # --- 确保传递给 vae_decode 的 latent 在 VAE 设备上且类型正确 ---
                current_pixels_segment = vae_decode(latents_to_decode_this_iter.to(device=vae.device, dtype=vae.dtype), vae).float().cpu() # 解码到 float32 CPU

                # --- 转换为 NumPy uint8 格式 [T, H, W, C] for imageio ---
                pixels_np = current_pixels_segment.squeeze(0) # 移除 Batch 维度 [C, T, H, W]
                pixels_np = pixels_np.permute(1, 2, 3, 0)     # [T, H, W, C]
                pixels_np = (pixels_np * 127.5 + 127.5).clamp(0, 255).byte() # [0, 255] uint8
                pixels_np = pixels_np.numpy() # 转 NumPy

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
                gc.collect() # 手动触发垃圾回收
                print("Memory released after decoding and writing.")
                # --- 结束关键步骤 ---

                if not high_vram:
                    unload_complete_models(vae) # 及时卸载 VAE

            # --- 更新 Gradio 界面 ---
            print(f'Decoded and wrote segment. Total latent frames generated: {total_generated_latent_frames}')
            if video_writer is not None:
                stream.output_queue.push(('file', output_filename)) # 推送文件名

            if is_last_section:
                break # 结束循环

        # --- 循环正常结束后，关闭写入器 ---
        if video_writer is not None:
            try:
                print("Closing video writer after loop completion...")
                video_writer.close()
                print("Video writer closed successfully after loop completion.")
                video_writer = None # 标记为已关闭
            except Exception as e_close:
                print(f"Error closing video writer after loop completion: {e_close}")

    except KeyboardInterrupt: # 处理用户中断
        print("User interrupted the process.")
        if video_writer is not None:
            try: video_writer.close(); print("Video writer closed after interruption.")
            except Exception as e: print(f"Error closing video writer after interruption: {e}")
            # video_writer = None

    except Exception: # 处理其他异常
        traceback.print_exc()
        if video_writer is not None:
            try: video_writer.close(); print("Video writer closed after exception.")
            except Exception as e: print(f"Error closing video writer after exception: {e}")
            # video_writer = None

    finally: # 最终清理
        print("Entering finally block...")
        # --- 卸载模型 ---
        if not high_vram:
            print("Unloading models in finally block...")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        # --- 发送结束信号 ---
        print("Pushing end signal to stream.")
        stream.output_queue.push(('end', None))
        print("Worker function finished.")

# --- Gradio 界面处理函数 ---
def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
    global stream
    assert input_image is not None, 'No input image!'
    print("Starting generation process...")

    # 清空上一次的输出并禁用开始按钮，启用结束按钮
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream() # 创建新的异步流

    # 异步运行 worker 函数
    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache)

    output_filename = None

    # 循环处理 worker 函数通过 stream 发送的数据
    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file': # 更新视频文件路径
            output_filename = data
            # 返回更新后的视频路径，其他 UI 组件保持不变或按需更新
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        elif flag == 'progress': # 更新进度条和预览
            preview, desc, html = data
            # 返回当前视频路径，更新预览图、描述和进度条
            yield gr.update(value=output_filename), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        elif flag == 'end': # 结束处理
            print("Generation process ended.")
            # 最终更新：显示最终视频，隐藏预览，清空进度，启用开始按钮，禁用结束按钮
            yield output_filename, gr.update(visible=False), gr.update(value=''), '', gr.update(interactive=True), gr.update(interactive=False)
            break # 退出循环

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

css = make_progress_bar_css() # 获取进度条 CSS
block = gr.Blocks(css=css).queue() # 创建 Gradio Blocks 界面，启用队列
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

            with gr.Group(): # 将高级选项分组
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # 负向提示词 (当前版本未使用)
                seed = gr.Number(label="Seed", value=31337, precision=0) # 随机种子
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1) # 视频总时长
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # 潜在空间窗口大小 (不建议修改)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.') # 采样步数
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # CFG Scale (当前版本未使用)
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.') # 蒸馏 CFG Scale
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # CFG Re-Scale (当前版本未使用)
                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.") # GPU 推理保留内存

        with gr.Column(): # 输出列
            preview_image = gr.Image(label="Sampling Preview (Middle Frame)", height=200, visible=False) # 预览图像
            result_video = gr.Video(label="Generated Video", autoplay=True, show_share_button=False, height=512, loop=True) # 结果视频
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation') # 进度描述
            progress_bar = gr.HTML('', elem_classes='no-generating-animation') # 进度条 HTML

    # 定义按钮点击事件
    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process) # 结束按钮调用 end_process

# --- 启动 Gradio 应用 ---
block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
