import os
import argparse

from mutagen.mp3 import MP3
from utils.img2mp4_utils import getRandomNumber, image_load, make_mp4, make_video
from txt2img_copy import txt2img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        help="문서를 작성하세요",
        default="What a nice day"
    )
    parser.add_argument(
        "--article",
        type=str,
        help="주제를 설정하세요",
        default="article",
    )
    parser.add_argument(
        "--chatgpt",
        type=str,
        help="chatgpt api 사용 유무를 입력하세요. T/F",
        default=False,
    )
    parser.add_argument(
        "--txt_file",
        type=str,
        help="txt 파일 이름을 입력하세요.",
        default="None"
    )

    opt = parser.parse_args()

    article = opt.article
    outpath = "./output/" + article
    chatgpt = opt.chatgpt
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    else:
        outpath = outpath + str(len(os.listdir("./output/")))
        os.makedirs(outpath)


    if opt.txt_file:
        contents_dict = {}
        with open("example.txt", "r") as f:
            for i, line in enumerate(f.readlines()):
                contents_dict[i] = line.replace("\n", "")
    else:
        contents_dict = {"prompt":opt.prompt}

    # video 생성
    for key, item in contents_dict.items():
        if type(key) == int:
            key = str(key)

        if type(item) == dict:
            for dict_key, dict_item in item.items():
                make_video(dict_key, dict_item, article, outpath, chatgpt)
        
        elif type(item) == list:
            item = ''.join(item)
            make_video(key, item, article, outpath, chatgpt)
        
        else:
            make_video(key, item, article, outpath, chatgpt)


    # sudo ffmpeg -f concat -safe 0 -i ./output/dog3/dog.txt -c copy ./output/dog3/output.mp4
    print(f"동영상 합성 : sudo ffmpeg -f concat -safe 0 -i {outpath}/{article}.txt -c copy {outpath}/output.mp4")

if __name__ == "__main__":
    main()
