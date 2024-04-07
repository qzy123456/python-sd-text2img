# Text2img
- Stable diffusion과 OPEN AI api를 이용하여 자동 동영상 생성
- img2mp4.py command 실행 후 나오는 동영상 합성 명령어에 따라 실행하면 생성 이미지와 mp3가 합성된 동영상이 만들어짐

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

python img2mp4.py -- article “Dog” -- txt_file “example.txt” --  prompt “happy,dog,cow,castle”
-- prompt “생성 이미지에 대한 설명” <br/>
-- article “주제”<br/>
-- chatgpt “T/F”<br/>
-- txt_file “txt 파일”<br/>
* txt_file이 있는 경우 prompt 실행 x

video 생성
site_video.py 파일 실행 -> 인터넷에서 동영상 파일을 가져옴
create_video.py 파일 실행 -> site_video로 생성된 파일을 이용하여 tts 및 영상 제작


