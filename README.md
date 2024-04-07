# Text2img


# 1. 환경빌드
conda env reate -f environment.yaml <br/>
conda activate ldm <br/>

https://github.com/CompVis/stable-diffusion#weights <br/>
Stable diffusion model downalod: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original <br/>

# 2. 환경
Python : 3.8.5<br/>
OS : Ubuntu 20.04<br/>
GPU : RTX A5000 24GB(최소 10GB이상 필요)<br/>
CUDA VERSION : 12.0<br/>

# 3. 실행 커맨드
python txt2img.py --prompt "I love puppies because so cute" --plms --outdir "dogandfood" -- chatgpt True

-- prompt “생성 이미지에 대한 설명”<br/>
-- plms plms 샘플링 방법<br/>
-- outdir “저장할 파일 이름”<br/>
-- chatgpt T/F(default = False) chatgpt api 사용 유무
