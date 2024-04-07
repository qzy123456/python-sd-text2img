import requests
import json
from urllib import parse 
import os
import sys
import urllib.request


API_KEY = 'PIXA Image api key'
query="고양이" #검색어
count=10 # 이미지 동영상 숫자
image_type="all" #Accepted values: "all", "photo", "illustration", "vector"
video_type="all"
outpath = "./script/video"
multi = ["강아지"]


def pixa_save_image(query, count, path):
    """
    query : 다운받고자하는 명령어
    count : 다운 받을 query의 개수
    path : 저장할 곳
    """
    path = os.path.join(path, query)
    if not os.path.exists(path):
            os.makedirs(path)
    else:
        print('폴더가 존재합니다')

    url3 = f'https://pixabay.com/api/?key={API_KEY}&q={query}&image_type=photo&per_page={count}'
    res = requests.get(url3)
    text= res.text

    d = json.loads(text)

    success=0
    for k in range(0,count):
        imgUrl=d['hits'][k]['webformatURL']
        url = parse.urlparse(imgUrl) 
        name, ext = os.path.splitext(url.path)

        filename = f'{query}_{k+1}{ext}'      
        saveUrl = os.path.join(path, filename)#저장 경로 결정    

        #파일 저장   
        req = urllib.request.Request(imgUrl, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            imgUrl = urllib.request.urlopen(req).read() #웹 페이지 상의 이미지를 불러옴
            with open(saveUrl,"wb") as f: #디렉토리 오픈
                f.write(imgUrl) #파일 저장  
            print(f"{saveUrl}에 저장 성공")
            success+=1         

        except urllib.error.HTTPError:
            print('에러')
            sys.exit(0)

    print('다운로드 성공 : '+ str(success))



def pixa_save_video(query, count, path):
    """
    query : 다운받고자하는 명령어
    count : 다운 받을 query의 개수
    path : 저장할 곳
    """
    path = os.path.join(path, query)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('폴더가 존재합니다')

    url3 = f'https://pixabay.com/api/videos/?key={API_KEY}&q={query}&veido_type={video_type}'
    res = requests.get(url3)
    res_text = res.text

    d = json.loads(res_text)

    success=0
    for k in range(0,count):
        video_url=d["hits"][k]["videos"]['large']['url']
        url = parse.urlparse(video_url) 
        name, ext = os.path.splitext(url.path)


        filename = f'{query}_{k+1}{ext}'      
        saveUrl = os.path.join(path,filename) #저장 경로 결정    
        print(saveUrl)

        #파일 저장   
        req = urllib.request.Request(video_url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            videoUrl = urllib.request.urlopen(req).read() #웹 페이지 상의 동영상을 불러옴
            with open(saveUrl,"wb") as f: #디렉토리 오픈
                f.write(videoUrl) #파일 저장  
            print(f"{saveUrl}에 저장 성공")
            success+=1         

        except urllib.error.HTTPError:
            print('에러')
            sys.exit(0)

    print('다운로드 성공 : '+str(success))


for query in multi:
    # pixa_save_image(query, count, path)
    pixa_save_video(query, count, outpath)

