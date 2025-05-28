import os 
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from tqdm.auto import tqdm
from transformers import pipeline
import numpy as np
import json
import time 

SWITCH_LORA_TIMESTEP = 800

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="input dataset")
    parser.add_argument("--output_dir",type=str,required=True,help="output directory for inpainted images")
    parser.add_argument("--index_file", type=str, default="", help="if none, we will get file from os.listdir instead")
    parser.add_argument("--seed", type=str, default="100,200,300,400,500", help="seed for generate, if there is more 1 seed, output_dir will automatically change to output_dir/seed")
    parser.add_argument("--evs", type=str,  help="EVs to use, default is 0,-2.5,-5", default="0,-2.5,-5")
    parser.add_argument("--prompt", type=str, default="a perfect mirrored reflective chrome ball sphere", help="prompts for inpainting, default is 'a perfect mirrored reflective chrome ball sphere'")
    parser.add_argument("--prompt_black", type=str, default="a perfect black dark mirrored reflective chrome ball sphere", help="black prompts for inpainting, default is 'a perfect black dark mirrored reflective chrome ball sphere'")
    parser.add_argument("--negative_prompt", type=str, default="matte, diffuse, flat, dull", help="negative prompts for inpainting, default is 'matte, diffuse, flat, dull'")
    parser.add_argument("--index", type=int, default=0, help="index of the image to inpaint, default is 0")
    parser.add_argument("--total", type=int, default=1, help="total number of images to inpaint, default is 1")
    parser.add_argument("--disable_lora", type=int, default=0, help="disable lora to measure the baseline")
    parser.add_argument("--measure_time", type=int, default=0, help="measure running time")
    return parser

def pil_square_image(image, desired_size = (1024,1024), interpolation=Image.LANCZOS):
    """
    Make top-bottom border
    """
    # Don't resize if already desired size (Avoid aliasing problem)
    if image.size == desired_size:
        return image
    
    # Calculate the scale factor
    scale_factor = min(desired_size[0] / image.width, desired_size[1] / image.height)

    # Resize the image
    resized_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)), interpolation)

    # Create a new blank image with the desired size and black border
    new_image = Image.new("RGB", desired_size, color=(0, 0, 0))

    # Paste the resized image onto the new image, centered
    new_image.paste(resized_image, ((desired_size[0] - resized_image.width) // 2, (desired_size[1] - resized_image.height) // 2))
    
    return new_image

def setup_sd():
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.load_lora_weights("DiffusionLight/Flickr2K", adapter_name="turbo")
    pipe.load_lora_weights("DiffusionLight/DiffusionLight", adapter_name="exposure")
    pipe.is_exposure_lora_loaded = False
    return pipe

def setup_depth_estimator():
    depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large", device="cpu")
    return depth_estimator

# create mask and depth map with mask for inpainting
def get_circle_mask(size=256):
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(1, -1, size)
    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0
    return mask 

def apply_lora(pipe, adapter_name, lora_scale):
    pipe.unfuse_lora()     # clear previous lora weights (if any)
    pipe.set_adapters(adapter_name)
    pipe.fuse_lora(lora_scale=lora_scale)

@torch.inference_mode()
def main():
    parser = create_argparser()
    args = parser.parse_args()

    evs = [float(ev.strip()) for ev in args.evs.split(',')]
    lowest_ev = min(evs)    
    highest_ev = max(evs)       

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chmod(args.output_dir, 0o777)
    except:
        pass

    # Check if the input image is 1024x1024
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Input image {args.input_image} does not exist.")

    if args.index_file != "":
        with open(args.index_file, 'r') as f:
            index_list = json.load(f)
    else:
        index_list = os.listdir(args.dataset)

    index_list = index_list[args.index::args.total]  # distribute across processes

    seeds = [int(s.strip()) for s in args.seed.split(',')]
    print("creating queues...")

    # create output directory for each seed (if need):
    if len(seeds) > 1:
        for seed in seeds:
            output_dir = os.path.join(args.output_dir, f"seed{seed}")
            os.makedirs(output_dir, exist_ok=True)
            try:
                os.chmod(output_dir, 0o777)
            except:
                pass
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        try:
            os.chmod(args.output_dir, 0o777)
        except:
            pass


    queues = []
    for index in tqdm(index_list):
        input_image_path = os.path.join(args.dataset, index)
        if not os.path.isfile(input_image_path):
            continue
        current_queue = []
        for seed in seeds:
            output_dir = os.path.join(args.output_dir, f"seed{seed}") if len(seeds) > 1 else args.output_dir            
            
            for ev_id, ev in enumerate(evs):
                file_front, file_ext = os.path.splitext(index)
                file_names = file_front.split('/')
                file_name = file_names[-1]
                ev_name = int(np.abs(ev) * 10)
                if len(file_names) > 1:
                    front_part = '/'.join(file_names[:-1])
                    output_image_path = os.path.join(output_dir, front_part, "raw", f"{file_name}_ev-{ev_name:02d}.png")
                else:
                    front_part = ''
                    output_image_path = os.path.join(output_dir, "raw", f"{file_name}_ev-{ev_name:02d}.png")
                if os.path.exists(output_image_path):
                    continue
                current_queue.append({
                    'ev_id': ev_id, 
                    'out_path': output_image_path,
                    'seed': seed,
                })
        queues.append({
            'input_index': index,
            'ev_queues': current_queue,
        })

    
    pipe = setup_sd()
    depth_estimator = setup_depth_estimator()
    
    # create prompt embeddings
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(args.prompt)
    prompt_embeds_black, _, pooled_prompt_embeds_black, _ = pipe.encode_prompt(args.prompt_black)
    negative_embeds, _, pooled_negative_embeds, _ = pipe.encode_prompt(args.negative_prompt) 

    # do pre-interpoaltion of prompts 
    prompt_embeds_interpolated = []
    pooled_prompt_embeds_interpolated = []
    for ev in evs:  
        ratio = (ev - lowest_ev) / (highest_ev - lowest_ev)
        prompt_embeds_interpolated.append(prompt_embeds * ratio + prompt_embeds_black * (1 - ratio))
        pooled_prompt_embeds_interpolated.append(pooled_prompt_embeds * ratio + pooled_prompt_embeds_black * (1 - ratio))

    # prepare mask
    depth_inner_mask = get_circle_mask(256).numpy()
    inpaint_mask = np.zeros((1024, 1024), dtype=np.float32)
    inpaint_mask_inner = get_circle_mask(256 + 20).numpy() 
    inpaint_mask[384 - 10:640 + 10, 384 - 10:640+10] = inpaint_mask_inner * 255
    inpaint_mask = Image.fromarray(inpaint_mask)

    # distribute queues across processes
    #queues = queues[args.index::args.total] 

    # callback function
    def callback(pipeline, i, t, callback_kwargs):
        # t will start from 999 and decrease to 0, we only activate once at t=800
        if not pipe.is_exposure_lora_loaded and t <= SWITCH_LORA_TIMESTEP:
            apply_lora(pipeline, "exposure", lora_scale=0.75)
            pipe.is_exposure_lora_loaded = True
        return callback_kwargs

    total_time = []
    with tqdm(total=len(queues), desc="Chromeball") as p:
        for queue in queues:
            p.update(1)
            if len(queue['ev_queues']) == 0:
                continue
            
            # preprocess input image
            input_image_path = os.path.join(args.dataset, queue['input_index'])
            input_image = Image.open(input_image_path).convert("RGB")
            input_image = pil_square_image(input_image, desired_size=(1024, 1024))

            # compute depth map with chrome ball mask
            depth_image = depth_estimator(images=input_image)['depth']
            depth = np.asarray(depth_image).copy()
            depth[384:640, 384:640] = depth[384:640, 384:640] * (1 - depth_inner_mask) + (depth_inner_mask * 255)
            depth_image = Image.fromarray(depth)

            for ev_queue in queue['ev_queues']:
                # DiffusionLight is sensitive to seed, seed should be same across EVs
                ev_id = ev_queue['ev_id']
                seed = ev_queue['seed']
                output_image_path = ev_queue['out_path']
                if os.path.exists(output_image_path):
                    continue
                ev = evs[ev_id]

                # make directory if needed
                output_image_dir = os.path.dirname(output_image_path)     
                 
                os.makedirs(output_image_dir, exist_ok=True)
                try:
                    os.chmod(output_image_dir, 0o777)
                except:
                    pass
                if not args.disable_lora:
                    apply_lora(pipe, "turbo", lora_scale=1.0)
    
                pipe.is_exposure_lora_loaded = False
                p.set_postfix_str(f"seed={seed}, ev={ev:.2f}")
                generator = torch.Generator(device="cuda").manual_seed(seed)
                start_time = time.time()
                callback_fn = callback if not args.disable_lora else None
                output = pipe(
                    prompt_embeds=prompt_embeds_interpolated[ev_id],
                    pooled_prompt_embeds=pooled_prompt_embeds_interpolated[ev_id],
                    negative_embeds=negative_embeds,
                    pooled_negative_embeds=pooled_negative_embeds,
                    num_inference_steps=30,
                    image=input_image,
                    mask_image=inpaint_mask,
                    control_image=depth_image,
                    controlnet_conditioning_scale=0.5,
                    callback_on_step_end=callback_fn,
                    generator=generator
                )
                if args.measure_time:
                    end_time = time.time()
                    total_time.append(end_time - start_time)
                    print(f"Processing {input_image_path} with seed={seed}, ev={ev:.2f} took {end_time - start_time:.2f} seconds")
                output_image = output.images[0]
                output_image.save(output_image_path)
                try:
                    os.chmod(output_image_path, 0o777)
                except:
                    pass

                square_image = output_image.crop((384, 384, 640, 640))
                square_dir = output_image_dir + "/../square"
                square_path = os.path.join(square_dir, f"{os.path.basename(output_image_path)}")
                os.makedirs(square_dir, exist_ok=True)
                square_image.save(square_path)
                try:
                    os.chmod(square_dir, 0o777)
                    os.chmod(output_image_path, 0o777)
                    os.chmod(square_path, 0o777)
                except:
                    pass
    
    if args.measure_time:
        print(f"Processed {len(total_time)} images.")
        if len(total_time) > 0:
            print(f"Average time per image: {np.mean(total_time):.6f} seconds")
            print(f"Max time: {max(total_time):.6f} seconds, Min time: {min(total_time):.6f} seconds")
            print("==============================")
            print(f"Total time: {sum(total_time):.6f} seconds")
            print(f"Average time per image: {np.mean(total_time) * len(evs):.6f} seconds")

        else:
            print("No images processed.")


if __name__ == "__main__":
    main()