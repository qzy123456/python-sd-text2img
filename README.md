# Text2img


# 1. 환경빌드
conda env reate -f environment.yaml

conda activate ldm

https://github.com/CompVis/stable-diffusion#weights

Stable diffusion model downalod: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original

# 2. 환경
Python : 3.8.5
OS : Ubuntu 20.04
GPU : RTX A5000 24GB(최소 10GB이상 필요)
CUDA VERSION : 12.0

# 3. 실행 커맨드
python txt2img.py --prompt "I love puppies because so cute" --plms --outdir "dogandfood" -- chatgpt True

-- prompt “생성 이미지에 대한 설명”

-- plms plms 샘플링 방법

-- outdir “저장할 파일 이름”

-- chatgpt T/F(default = False) chatgpt api 사용 유무
