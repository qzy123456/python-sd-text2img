import cv2
from mutagen.mp3 import MP3
import urllib.request
import ffmpeg
import random
import os
import gc
from random import randint
from txt2img import txt2img

clova_client_id = "CLOVA_CLIENT_ID"
clova_client_secret = "CLOVA_CLIENT_SECRET"
clova_url = "CLOVA_URL"


width = 1920
height = 1080
fps = 1
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
screenshoot_length = 15
screen_stay_seconds = 3


def getRandomNumber(arr):
    return randint(0, len(arr)-1)



def image_load(dir_path):
    img_path = []
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            if '.png' in file:
                file_path = os.path.join(root, file)
                img_path.append(file_path)

    return img_path
    


def make_mp4(key, item, article, outpath, img_path):
    """
    key : ditionary
    item : 구문
    article : 주제
    outpath : 저장 공간
    img_path : 입힐 이미지
    """
    print(f"{article}의 {key} | {item} voice start ")
    file_mp3 = outpath + f"/{key}/{key}.mp3"
    file_avi = outpath + f"/{key}/{key}.mp4"

    encText = urllib.parse.quote(item) # 구문
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

    audio = MP3(file_mp3)
    audio_length = round(audio.info.length)
    out = cv2.VideoWriter(file_avi, fourcc, fps, (width, height)) 
    seq = 0

    for i in range(audio_length):
        if (i % screen_stay_seconds) == 0:
            file_png = random.choice(img_path) # 이미지 가져오는 곳
            print(str(i), file_png, str(width), str(height))
            frame = cv2.imread(file_png)
            frame = cv2.resize(frame, (width, height))
            if seq < (screenshoot_length - 1):
                seq += 1
            else:
                seq = screenshoot_length - 1
        for i in range(fps):
            out.write(frame)
    out.release()
    
    file_mp4 = outpath + f"/{key}/{key}_final.mp4"
    with open(outpath + f"/{article}.txt", 'a') as f:
        file_name = "file " + f"./{key}/{key}_final.mp4" + "\n"
        f.write(file_name)

    video = ffmpeg.input(file_avi)
    audio = ffmpeg.input(file_mp3)
    out = ffmpeg.output(video, audio, file_mp4)
    out.run()
    
    del out, video, audio
    gc.collect()
    print(f"{article}의 {key} | {item} voice end")
    print("=====================================")




def make_video(key, item, article, outpath, chatgpt):
    txt2img(key, item, article, outpath, chatgpt = chatgpt) # txt -> img 생성(stable_diffusion)
    img_path = image_load(os.path.join(outpath, key) + "/")
    make_mp4(key, item, article, outpath, img_path) # 생성된 이미지를 mp4(video로 생성)

