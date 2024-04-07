from urllib import request
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import requests
import json
import urllib3


access_key = 'UNSPLASH API KEY'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def img_downloads(query, count, W, H):
    """
    query : 가져올 이미지를 위한 query문
    count : 가져올 이미지 개수
    H : 이미지의 높이
    W : 이미지의 넓이
    return : 이미지 리스트
    """
    url = 'https://api.unsplash.com/photos/random?' + 'client_id=' + access_key
    imgs_list = []

    parameters = {
        'query': query,
        'count': count,
    }

    response = requests.get(url, params=parameters, verify=False)
    if response.status_code != 200:
        raise print("가져오지 못했습니다.")
    json_object = json.loads(response.text)
    for i in range(int(count)):
        url = json_object[i]['urls']['small']
        res = request.urlopen(url).read()
        img = Image.open(BytesIO(res))
        numpy_img = np.array(img)
        img_h, img_w = numpy_img.shape[:2]
        
        if img_h == H and img_w == W:
            imgs_list.append(numpy_img)
        elif img_h > H and img_w > W:
            img_scaled = cv2.resize(numpy_img, (W, H), interpolation=cv2.INTER_AREA) # 이미지를 축소할 때 사용하는 보간법
            imgs_list.append(img_scaled)
        else:    
            img_scaled = cv2.resize(numpy_img,(W, H),interpolation=cv2.INTER_LINEAR) # 이미지를 키울 때 사용하는 보간법
            imgs_list.append(img_scaled)

    return imgs_list



def mirroring(img):
    img = cv2.flip(img, 1)
    return img



def flip(img):
    img = cv2.flip(img, 0)
    return img



def rotate(img):
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 회전
    return img


def random_img_list(img_list):
    """
    input : 이미지 리스트
    output : 이미지 리스트에 있는 파일 중 0.5의 확률로 mirroring, flip, rotate를 거친 파일들
    """
    for i in range(len(img_list)):
        t_f = random.choice([0,1])
        if t_f == 1:
            img_list[i] = mirroring(img_list[i])
        
        t_f = random.choice([0,1])
        if t_f == 1:
            img_list[i] = flip(img_list[i])
        
        t_f = random.choice([0,1])
        if t_f == 1:
            img_list[i] = rotate(img_list[i])
    
    random.shuffle(img_list)

    return img_list
