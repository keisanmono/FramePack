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
# import safetensors.torch as sf
import numpy as np
import argparse
import math
import imageio
import gc

# --- 导入 Accelerate ---
from accelerate import Accelerator
from accelerate.utils import release_memory

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
# 确保导入了所有需要的辅助函数
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake, DEFAULT_PROMPT_TEMPLATE # <-- 确保导入 DEFAULT_PROMPT_TEMPLATE
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
# --- 移除手动内存管理相关的导入 ---
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
if torch.cuda.is_available():
    gpu = torch.device("cuda:0")
    torch.cuda.empty_cache()
else:
    gpu = torch.device("cpu")
print(f"Primary compute device selected: {gpu}")


# --- 使用 Accelerate 加载支持 device_map 的模型，手动加载不支持的 ---
print("Loading models...")
models = {}
try:
    print("Loading text_encoder with device_map='auto'...")
    models['text_encoder'] = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True
    )
    gc.collect(); torch.cuda.empty_cache()
    print("Loading text_encoder_2 with device_map='auto'...")
    models['text_encoder_2'] = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True
    )
    gc.collect(); torch.cuda.empty_cache()
    print("Loading Transformer to CPU (device_map not supported)...")
    models['transformer'] = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        'lllyasviel/FramePackI2V_HY', torch_dtype=torch.float16
    ).cpu()
    gc.collect(); torch.cuda.empty_cache()
    print("Loading VAE to CPU (device_map not supported)...")
    models['vae'] = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16
    ).cpu()
    gc.collect(); torch.cuda.empty_cache()
    print("Attempting to load image_encoder with device_map='auto'...")
    try:
        models['image_encoder'] = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True
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
    compute_device = gpu

    transformer_on_gpu = False
    vae_on_gpu = False
    image_encoder_on_gpu = False

    try:
        # --- 文本编码 ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        print("Starting text encoding...")
        # 调用编码函数获取 positive prompt 的 embeddings
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        # 需要获取对应的 attention mask
        prompt_llama_pos = [DEFAULT_PROMPT_TEMPLATE["template"].format(prompt)]
        crop_start = DEFAULT_PROMPT_TEMPLATE["crop_start"]
        llama_inputs_pos = tokenizer(prompt_llama_pos, padding="max_length", max_length=512 + crop_start, truncation=True, return_tensors="pt", return_attention_mask=True)
        llama_attention_mask = llama_inputs_pos.attention_mask[:, crop_start:llama_vec.shape[1]+crop_start] # 调整 mask 长度以匹配 llama_vec

        # --- [修正] 初始化负向变量并处理 ---
        llama_vec_n = None
        clip_l_pooler_n = None
        llama_attention_mask_n = None

        if cfg == 1:
            print("CFG=1, using zero negative embeddings.")
            llama_vec_n = torch.zeros_like(llama_vec)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
            llama_attention_mask_n = torch.ones_like(llama_attention_mask) # 创建全1的 mask
        else:
            print("CFG!=1, encoding negative prompt...")
            # 假设 encode_prompt_conds 只返回 vec 和 pooler, 需要再次调用处理负向提示
            # (如果 encode_prompt_conds 能处理 negative 就更好了)
            # 这里我们模拟调用两次
            if n_prompt is None or n_prompt.strip() == "":
                # 如果负向提示为空，也使用零向量
                print("Empty negative prompt, using zero negative embeddings.")
                llama_vec_n = torch.zeros_like(llama_vec)
                clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
                llama_attention_mask_n = torch.ones_like(llama_attention_mask)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                # 获取负向 mask
                prompt_llama_neg = [DEFAULT_PROMPT_TEMPLATE["template"].format(n_prompt)]
                llama_inputs_neg = tokenizer(prompt_llama_neg, padding="max_length", max_length=512 + crop_start, truncation=True, return_tensors="pt", return_attention_mask=True)
                llama_attention_mask_n = llama_inputs_neg.attention_mask[:, crop_start:llama_vec_n.shape[1]+crop_start] # 调整 mask 长度

            print("Negative prompt encoded.")
        # --- [结束 修正] ---
        print("Text encoding finished.")


        # --- 处理输入图像 ---
        # ... (不变) ...
        input_image_pt = input_image_pt.cpu() # 确保在 CPU

        # --- VAE 编码 ---
        # ... (不变: to(gpu), encode, to(cpu)) ...
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        print("Moving VAE to GPU for encoding...")
        vae.to(gpu); vae_on_gpu = True
        gc.collect(); torch.cuda.empty_cache()
        print("VAE moved to GPU.")
        print("Starting VAE encoding...")
        start_latent = vae_encode(input_image_pt.to(device=vae.device, dtype=vae.dtype), vae)
        print(f"VAE encoding finished...")
        print("Moving VAE back to CPU...")
        vae.to(cpu); vae_on_gpu = False
        gc.collect(); torch.cuda.empty_cache()
        print("VAE moved to CPU.")
        start_latent = start_latent.to(compute_device)
        print(f"Start latent moved to {compute_device}")


        # --- CLIP Vision 编码 ---
        # ... (不变: 检查是否在 CPU, to(gpu), encode, to(cpu)) ...
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        if hasattr(image_encoder, 'device') and image_encoder.device == torch.device('cpu'):
            print("Moving Image Encoder to GPU for encoding...")
            image_encoder.to(gpu); image_encoder_on_gpu = True
            gc.collect(); torch.cuda.empty_cache()
            print("Image Encoder moved to GPU.")
        print("Starting CLIP Vision encoding...")
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        print(f"CLIP Vision encoding finished...")
        if image_encoder_on_gpu:
             print("Moving Image Encoder back to CPU...")
             image_encoder.to(cpu); image_encoder_on_gpu = False
             gc.collect(); torch.cuda.empty_cache()
             print("Image Encoder moved to CPU.")
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(compute_device)
        print(f"Image encoder hidden state moved to {compute_device}")


        # --- Dtype 转换 ---
        print("Converting dtypes and moving tensors...")
        start_latent = start_latent.to(torch.float16)
        llama_vec = llama_vec.to(compute_device, dtype=torch.float16)
        clip_l_pooler = clip_l_pooler.to(compute_device, dtype=torch.float16)
        # --- [修正] 确保转换前非 None (虽然上面逻辑已保证) ---
        llama_vec_n = llama_vec_n.to(compute_device, dtype=torch.float16)
        clip_l_pooler_n = clip_l_pooler_n.to(compute_device, dtype=torch.float16)
        # --- [结束 修正] ---
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(compute_device, dtype=torch.float16)
        print("Embeddings and poolers moved and converted to float16 on compute device.")


        # --- 将 Transformer 移到 GPU (在采样循环开始前) ---
        print("Moving Transformer to GPU for sampling...")
        transformer.to(gpu); transformer_on_gpu = True
        gc.collect(); torch.cuda.empty_cache()
        print("Transformer moved to GPU.")
        # --- 结束 ---


        # --- Sampling (采样) ---
        # ... (rnd, num_frames, history_latents, latent_paddings 初始化不变) ...

        # --- 主采样循环 ---
        for latent_padding in latent_paddings:
            # ... (循环开始和索引计算不变) ...
            # ... (clean_latents 计算不变) ...

            # --- 定义回调函数 ---
            # ... (不变) ...
            def callback(d):
                try:
                    preview = d['denoised']
                    preview_vae_device = gpu
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

            # --- 执行采样 ---
            print("Starting sampling...")
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
                prompt_embeds=llama_vec.to(compute_device),
                # --- [修正] 确保 mask 在 compute_device ---
                prompt_embeds_mask=llama_attention_mask.to(compute_device),
                prompt_poolers=clip_l_pooler.to(compute_device),
                negative_prompt_embeds=llama_vec_n.to(compute_device),
                negative_prompt_embeds_mask=llama_attention_mask_n.to(compute_device),
                # --- [结束 修正] ---
                negative_prompt_poolers=clip_l_pooler_n.to(compute_device),
                device=compute_device,
                dtype=torch.float16,
                image_embeddings=image_encoder_last_hidden_state.to(compute_device),
                latent_indices=latent_indices,
                clean_latents=clean_latents.to(device=compute_device, dtype=torch.float16),
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x.to(device=compute_device, dtype=torch.float16),
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x.to(device=compute_device, dtype=torch.float16),
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            print("Sampling finished for this section.")
            # generated_latents 在 compute_device (gpu) 上

            # --- 处理采样结果 ---
            # ... (不变) ...

            # --- VAE 解码和流式写入 ---
            # ... (不变: to(gpu), decode, to(cpu), write, del, gc.collect()) ...
            if added_latent_frames_count > 0:
                print("Moving VAE to GPU for decoding...")
                vae.to(gpu); vae_on_gpu = True
                gc.collect(); torch.cuda.empty_cache()
                print("VAE moved to GPU.")
                latents_to_decode_this_iter = history_latents[:, :, :added_latent_frames_count, :, :] # 从 history 切片 (在 CPU)
                current_pixels_segment = vae_decode(latents_to_decode_this_iter.to(device=vae.device, dtype=vae.dtype), vae).float().cpu() # 解码到 float32 CPU
                print("VAE decoding for writing finished.")
                print("Moving VAE back to CPU...")
                vae.to(cpu); vae_on_gpu = False
                gc.collect(); torch.cuda.empty_cache()
                print("VAE moved to CPU.")
                pixels_np = current_pixels_segment.squeeze(0)
                pixels_np = pixels_np.permute(1, 2, 3, 0)
                pixels_np = (pixels_np * 127.5 + 127.5).clamp(0, 255).byte()
                pixels_np = pixels_np.numpy()
                if video_writer is None:
                    fps_to_save = 30
                    video_writer = imageio.get_writer(output_filename, fps=fps_to_save, codec='libx264', quality=8, output_params=['-preset', 'fast', '-tune', 'fastdecode'], macro_block_size=16, ffmpeg_log_level='warning')
                    print(f"Video writer initialized for {output_filename}")
                print(f"Appending {pixels_np.shape[0]} frames to video...")
                for i in range(pixels_np.shape[0]):
                    video_writer.append_data(pixels_np[i])
                print(f"Frames appended.")
                del current_pixels_segment, pixels_np, latents_to_decode_this_iter
                gc.collect()
                print("Memory released after decoding and writing.")

            # --- 更新 Gradio 界面 ---
            # ... (不变) ...

            if is_last_section:
                break

        # --- 循环正常结束后，关闭写入器 ---
        # 这个逻辑移到下面的 finally 块中处理

    except KeyboardInterrupt: # 处理用户中断
        print("User interrupted the process.")

    except Exception: # 处理其他异常
        traceback.print_exc()

    finally: # 最终清理
        print("Entering final finally block...")
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
        if transformer_on_gpu:
            try:
                print("Moving Transformer back to CPU in finally...")
                transformer.to(cpu)
            except Exception as e_trans_unload: print(f"Error moving Transformer to CPU in finally: {e_trans_unload}")
        if vae_on_gpu: # 检查 VAE 是否还在 GPU (不太可能，但以防万一)
             try:
                 print("Moving VAE back to CPU in finally (final check)...")
                 vae.to(cpu)
             except Exception as e_vae_unload: print(f"Error moving VAE to CPU in finally: {e_vae_unload}")
        if image_encoder_on_gpu: # 检查 Image Encoder 是否还在 GPU
             try:
                 print("Moving Image Encoder back to CPU in finally (final check)...")
                 image_encoder.to(cpu)
             except Exception as e_img_unload: print(f"Error moving Image Encoder to CPU in finally: {e_img_unload}")

        # --- 尝试释放 Accelerate 管理的模型 ---
        try:
            print("Releasing models potentially managed by Accelerate...")
            release_memory(models.get('text_encoder'), models.get('text_encoder_2'))
            print("Attempting final garbage collection and cache clearing...")
            # 不再需要 del models[...]
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
def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache): # 移除 gpu_memory_preservation
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
