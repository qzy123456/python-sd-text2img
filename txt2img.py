import os
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from einops import rearrange
# from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.models.diffusion.ddim import DDIMSampler # Diffusion Deep Inference Machines 이미지 생성에 사용되며, 주어진 조건에 따라 점진적으로 확산 시켜서 이미지 생성
from ldm.models.diffusion.plms import PLMSSampler # Progressive Last State Model 이미지의 점진적인 생성 과정을 다양한 단계로 나누어 진행
from ldm.models.diffusion.dpm_solver import DPMSolverSampler # 마지막 상태 모델을 풀기 위한 샘플링 기능 제공
import transformers

from utils.txt2img_utils import *
from utils.make_prompt import make_stable_diffusion_text

transformers.logging.set_verbosity_error()
TF_ENABLE_ONEDNN_OPTS=0
import time

sampler_method = "plms"
fixed_code = True
ddim_eta = 0.0 # corressponds to deterministic sampling
n_iter = 2 # sample
H = 512 # image height, in pixel space
W = 512 # image width
C = 4 # latent channels
F = 8 # downsampling factor
n_samples = 3 # how many samples to produce for each given prompt. A.k.a. batch size
scale = 7.5 # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
from_file = None
model_config = "./configs/stable-diffusion/v1-inference.yaml" # 모델을 구성하는 yaml 경로
ckpt = "./models/stable-diffusion-v1/model.ckpt" # ckpt 경로
seed = 42 # seed
precision = "autocast" # full, autocast precision 평가 지표

seed_everything(seed) # seed 설정
config = OmegaConf.load(f"{model_config}")
model = load_model_from_config(config, f"{ckpt}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
print(f"CUDA USE: {torch.cuda.is_available()}")


if sampler_method == "dpm_solver":
    sampler = DPMSolverSampler(model)
elif sampler_method == "plms":
    sampler = PLMSSampler(model)
else:
    sampler = DDIMSampler(model)

print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
wm = "StableDiffusionV1"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    

def txt2img(key, prompt, article, outpath, n_iter = 2, n_samples = 3, chatgpt = False, seed = 42):
    """
    key : contents_dict의 key
    prompt : 입력받는 item
    article : 주제(저장되는 폴더명)
    outpath : 결과 저장 폴더
    output : prompt의 txt 파일과 생성된 파일
    """
    print(f"{article}의 {key} | {prompt} txt2img start ")
    start_time = time.time()

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    batch_size = n_samples


    sample_path = os.path.join(outpath, key)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    else:
        sample_path = sample_path + str(len(os.listdir(outpath)))
        os.makedirs(sample_path)

    if not from_file:
        if chatgpt:
            prompt = make_stable_diffusion_text(prompt, sample_path)
            print("CHATGPT PROMPT : {}".format(prompt))
        else:
            prompt = prompt
            print("NORMAL PROMPT : {}".format(prompt))
        assert prompt is not None
        data = [batch_size * [prompt]]


    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

   
    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // F, W // F], device=device)


    precision_scope = autocast if precision=="autocast" else nullcontext
    base_count = 0

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"): # trange 파이썬 진행 프로세스
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [C, H // F, W // F]
                        samples_ddim, _ = sampler.sample(S=50, # ddim sampling steps
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)


                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            print(f"이미지저장 {sample_path}" + "/" + f"{base_count:05}.png")
                            base_count += 1


                if not chatgpt: # prompt 저장
                        with open(sample_path + "/chatgptprompt.txt", 'w', encoding="UTF-8") as f:
                                    f.write(prompt)
    
    end_time = time.time()
    print(f"{end_time - start_time:.4f} sec") # 수행시간
    print(f"{article}의 {key} | {prompt} txt2img end ")
    print("=========================================")





