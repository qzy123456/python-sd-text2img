import os
from random import randint
import urllib
import random
import numpy as np
import re

import natsort # 비디오 파일 정렬
from moviepy.editor import VideoFileClip # 오디오, 비디오 파일 부르기
from moviepy.editor import concatenate_videoclips # 비디오 클립 합치기
from moviepy.editor import TextClip, CompositeVideoClip # 텍스트 클립 적용, 비디오 합치기
from moviepy.editor import transfx # slid_in에 사용
from moviepy.video.tools.drawing import circle # The end 효과에 사용

from moviepy.editor import ImageSequenceClip # Image 클립
from moviepy.editor import ImageClip
from moviepy.editor import VideoFileClip # Vedio 클립
from moviepy.editor import CompositeVideoClip # 구성하기
from PIL import ImageDraw, Image# Font를 사용하기 위함
from moviepy.video.tools.segmenting import findObjects # Object 찾기


clova_client_id = "pqggz5ggoc"
clova_client_secret = "YSDfQj9bIoONItx7Mc0roVxVALabJNHfhr6Jj6M2"
clova_url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"


def getRandomNumber(arr):
    return randint(0, len(arr)-1)


def make_tts(txt_path, input_filename): # 구글
    """
    txt_path : tts를 읽을 대본 경로
    txt : txt 이름
    file_mp3 : mp3 이름
    """
    with open(txt_path + input_filename, "r") as f:
        item = f.readlines()
    item = list(map(lambda x: x.replace("\n", ""), item))
    join_item = ''.join(item)

    file_mp3 = os.path.join(os.path.dirname(txt_path), input_filename.split(".")[0] + ".mp3") 
    encText = urllib.parse.quote(join_item) # 구문
    speakers = ['mijin', 'jinho', 'nara', 'nminsang', 'nhajun', 'ndain', 'njiyun', 'nsujin', 'njinho', 'nsinu', 'njihun'] # 말하는 사람
    speaker = speakers[getRandomNumber(speakers)] # 랜덤 스피커
    data = "speaker=" + speaker + "&volume=0&speed=0&pitch=0&format=mp3&text=" + encText # 구문 읽기
    request = urllib.request.Request(clova_url) # clova api
    request.add_header("X-NCP-APIGW-API-KEY-ID",clova_client_id)
    request.add_header("X-NCP-APIGW-API-KEY",clova_client_secret)
    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        with open(f"{file_mp3}", 'wb') as f:
            f.write(response_body)
            print(f"{file_mp3} : 완료")



def random_slide(first_vedio, second_vedio):
    """
    랜덤 슬라이드 효과 : (왼쪽, 오른쪽), (오른쪽, 왼쪽), (위, 아래), (아래, 위)
    -----------------------------------------
    first_vedio : 첫번째 비디오 클립(나갈 클립)
    second_vedio : 두 번째 비디오 클립(들어올 클립)
    slided_clips_concat : first_vedio와 second vedio가 합쳐진 클립
    """
    
    random_slide_para = [["left", "right"], ["right", "left"], ["top", "bottom"], ["bottom", "top"]]
    random_para = random.choice(random_slide_para)
    first, second = random_para[0], random_para[1]

    slided_clips = [
        CompositeVideoClip([first_vedio.fx(transfx.slide_out, first_vedio.duration, first), second_vedio.fx(transfx.slide_in, second_vedio.duration, second)])
    ]

    slided_clips_concat = concatenate_videoclips(slided_clips, padding=-1)

    return slided_clips_concat



def read_srt_file(srt_path):
    """
    https://huggingface.co/spaces/aadnk/whisper-webui srt파일 생성 whisper-web ui를 통해 자막 파일 생성
    srt 파일 읽기
    ====================================================================================
    1
    00:00:00,000 --> 00:00:09,440
    강아지를 위한 안전하고 편안한 공간 조성 안녕하세요. 
    강아지에게 안락하고 안전한 공간을 마련하는 것은 매우 중요합니다.

    2
    00:00:09,440 --> 00:00:23,440
    강아지가 자유롭게 휴식하고 안전하게 놀 수 있는 공간은 행복과 정서에 연결되기 때문입니다. 
    안전하고 편한 환경에서 자란 강아지는 사랑과 좋은 정서적인 감정을 나눌 수 있습니다.

    3
    00:00:23,440 --> 00:00:26,752
    좋은 휴식 공간을 통해 삶의 질을 향상하고
    ====================================================================================
    srt_path : srt 파일 경로
    subtitles : 아래의 형식을 가진 Dictonary 생성
    
    {number : 1,
    start_time : 00:00:00,000,
    out_time : 00:00:09,440,
    text : 강아지를 위한 안전하고 편안한 공간 조성 안녕하세요. 강아지에게 안락하고 안전한 공간을 마련하는 것은 매우 중요합니다.}

    """
    subtitles = []
    with open(srt_path, 'r') as file:
        lines = file.readlines()

    # SRT 파일은 자막 번호, 시간 정보, 텍스트로 구성되므로 각 자막을 파싱합니다.
    subtitle_number = None
    start_time = None
    end_time = None
    subtitle_text = ''
    for line in lines:
        line = line.strip()  # 공백 제거

        if subtitle_number is None:
            subtitle_number = int(line)
        elif start_time is None:
            start_time, end_time = line.split(' --> ')
        elif line == '':
            # 자막 정보를 모두 수집한 후, 자막 객체를 생성하고 리스트에 추가합니다.
            subtitle = {
                'number': subtitle_number,
                'start_time': start_time,
                'end_time': end_time,
                'text': subtitle_text
            }
            subtitles.append(subtitle)

            # 다음 자막을 위해 변수들을 초기화합니다.
            subtitle_number = None
            start_time = None
            end_time = None
            subtitle_text = ''
        else:
            # 텍스트를 계속 추가합니다.
            subtitle_text += line + ' '

    return subtitles



def input_text(frame, font, text):
    """
    frame : vedio_clip frame
    font : font
    text : frame에 넣을 text
    img : 대사를 삽입한 이미지
    """
    h, w = frame.shape[:2]
    
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    left, top, right, bottom = draw.textbbox((w/2,h*0.9), text, font=font, anchor="mt") # anchor 가운데 하단에 저장
    draw.rectangle((left-5, top-5, right+5, bottom+5), fill='black') # 박스 색깔 정할 수 있음
    draw.text((w/2,h*0.9), text, (255, 255, 255), font=font, anchor="mt") # h, w
    img = np.array(img_pil)

    return img


def sub_titles_transition(video_part_path):
    """
    input : 나누어진 vedio를 받음 
    1_shorts.mp4 
    정규표현식 : "\d+_shorts.mp4"
    sub_video : transition 후의 비디오
    """
    p = re.compile("\d+_shorts.mp4")  # 정규표현식
    padding = 2
    video_part_file = [os.path.join(video_part_path, shorts) for shorts in os.listdir(video_part_path) if p.match(shorts)]
    sorted_video_part_file = natsort.natsorted(video_part_file)
    video_clips = [VideoFileClip(video_file) for video_file in sorted_video_part_file]

    video_fx_list = [video_clips[0]]

    idx = video_clips[0].duration - padding
    for video in video_clips[1:]:
        video_fx_list.append(video.set_start(idx).crossfadein(padding))
        idx += video.duration - padding

    sub_video = CompositeVideoClip(video_fx_list).add_mask()
    
    return sub_video



def make_head(img, title, second, size, font):
    """
    입력한 img에 관해 영상 시작을 만들어 줌
    ------------------------------------
    img : 영상 시작 이미지
    title : 영상 이름
    second : 영상 초
    size : 영상 크기(w,h)
    font : 제목 폰트
    head_clip : 제작된 영상 시작 클립
    """
    title_image = ImageClip(img, ismask=True).add_mask()
    title_image = title_image.resize(size).to_RGB().set_duration(second)
    
    txtClip = TextClip("MOVIE SHORTS",color='white', font="Amiri-Bold",
                    kerning = 5, fontsize=100)
    cvc = CompositeVideoClip([txtClip.set_position('center')], size=size)

    rotMatrix = lambda a: np.array( [[np.cos(a),np.sin(a)], 
                                    [-np.sin(a),np.cos(a)]] )

    def vortex(screenpos,i,nletters): # 소용돌이
        d = lambda t : 1.0/(0.3+t**8) #damping
        a = i*np.pi/ nletters # angle of the movement
        v = rotMatrix(a).dot([-1,0])
        if i%2 : v[1] = -v[1]
        return lambda t: screenpos+400*d(t)*rotMatrix(0.5*d(t)*a).dot(v)
        
    def cascade(screenpos,i,nletters): # 폭포수
        v = np.array([0,-1])
        d = lambda t : 1 if t<0 else abs(np.sinc(t)/(1+t**4))
        return lambda t: screenpos+v*400*d(t-0.15*i)

    def arrive(screenpos,i,nletters): # 밖으로 떠나는
        v = np.array([-1,0])
        d = lambda t : max(0, 3-3*t)
        return lambda t: screenpos-400*v*d(t-0.2*i)
        
    def vortexout(screenpos,i,nletters): # 소용돌이 밖으로
        d = lambda t : max(0,t) #damping
        a = i*np.pi/ nletters # angle of the movement
        v = rotMatrix(a).dot([-1,0])
        if i%2 : v[1] = -v[1]
        return lambda t: screenpos+400*d(t-0.1*i)*rotMatrix(-0.2*d(t)*a).dot(v)


    letters = findObjects(cvc) # 개체 찾기를 통해 각 문자를 찾아 구분함
    
    def moveLetters(letters, funcpos):
        return [ letter.set_position(funcpos(letter.screenpos,i,len(letters)))
              for i,letter in enumerate(letters)]

    funcpos_list = [vortex, cascade, arrive, vortexout]

    clips = [ CompositeVideoClip( moveLetters(letters,random.choice(funcpos_list)), # vortex , cascade, arrive, vortexout
                                size = size).subclip(0,5) ] 

    # 파일 합성
    final_clip = concatenate_videoclips(clips)
    final_video_clip = CompositeVideoClip([title_image,final_clip])

    title_frame = []

    for t in range(final_video_clip.duration):
        frame = final_video_clip.make_frame(t)
        title_frame.append(input_text(frame, font, title))

    head_clip = ImageSequenceClip(title_frame, 1).add_mask() # 5장의 sequence 이미지 

    return head_clip


def make_end(video_file, text, end_second):
    """
    입력한 img의 끝 효과
    ------------------------------------
    video_file : 영상
    text : 영상 마무리 멘트
    end_second : 영상 끝날때 초
    end_clip : 제작된 영상 끝 클립
    """
    # end 효과
    clip = video_file.subclip(video_file.duration-end_second, video_file.duration).add_mask()

    w, h = clip.size

    # The mask is a circle with vanishing radius r(t) = 800-200*t
    clip.mask.get_frame = lambda t: circle(
        screensize=(clip.w, clip.h),
        center=(clip.w / 2, clip.h / 4),
        radius=max(0, int(800 - (800/clip.duration) * t)),
        col1=1,
        col2=0,
        blur=4,
    )

    the_end = TextClip(text, font="Amiri-bold", color="white", fontsize=70).set_duration(clip.duration)
    end_clip = CompositeVideoClip([the_end.set_position("center"), clip], size=clip.size)

    return end_clip
