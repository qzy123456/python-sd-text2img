from utils.create_video_utils import *
import psutil
import gc
import math
from PIL import ImageFont
from moviepy.editor import AudioFileClip
import glob

out_path = "./script/example/" # 결과 저장 경로
video_path = "./script/video" # 기존 비디오 파일 경로

img = "./title_image.jpg" # 표지 이미지
font = ImageFont.truetype('Tenada.ttf', 30) # 썸네일 폰트
title = "흥미로운 강아지 이야기" # 표지 제목 
second = 5 # head, end clip의 시간
size = (1920, 1080) # video size
text = "The end" # 마무리 멘트


make_tts(out_path, "example.txt") # tts에서 바론 대본으로 갈 수 있도록 하기
video_path_file = glob.glob(video_path +  "/*/*")

sorted_video_file = natsort.natsorted(video_path_file)
file_mp3 = out_path + "example.mp3"
audio_file = AudioFileClip(file_mp3)
# audio에 맞게 Video 생성
video_file = VideoFileClip(sorted_video_file[0])
count = 2


while video_file.duration <= audio_file.duration: 
    """
    동영상의 최소 길이 맞추는 작업
    video size 크기가 random하게 들어올 경우 일관된 형식으로 맞추는 작업이 필요함
    """
    video_file2 = VideoFileClip(sorted_video_file[count]) # 생성 이미지 불러오는 곳으로 바꿔야 함
    first_video = video_file.subclip(video_file.duration-5, video_file.duration).add_mask()
    second_video = video_file2.subclip(0, 5).add_mask()
    slide_concat_clips = random_slide(first_video, second_video)
    video_file = concatenate_videoclips([video_file.set_end(video_file.duration-5),slide_concat_clips, video_file2.set_start(5.5)]) # 동영상이 매끄럽게 결합되지 않음
    count += 1

process = psutil.Process()
memory_info = process.memory_info()
print("현재 메모리 사용량 (MB):", memory_info.rss / 1024 / 1024)


# audio에 맞게 Video 자르기
video_file = video_file.set_duration(math.ceil(audio_file.duration))

del first_video, second_video, slide_concat_clips, count, video_path_file
gc.collect()

# srt 대본 파일 읽기
subtitles = read_srt_file(out_path + "example.srt")

# srt 파일을 이용하여 각 파트 별로 저장
for i,subtitle in enumerate(subtitles):
    print("======================================")
    print(f"===={subtitle['number']} 동영상 시작====")
    new_frame = []
    text = subtitle["text"]
    sub_clip = video_file.subclip(subtitle["start_time"], subtitle["end_time"])
    
    for frame in sub_clip.iter_frames():
        new_frame.append(input_text(frame, font, text))
            
    clip = ImageSequenceClip(new_frame, fps=60).add_mask()
    # clips.append(clip)
    clip.write_videofile(f'{out_path}/{subtitle["number"]}_shorts.mp4',fps=clip.fps,codec='mpeg4')
    if i % 2 == 0:
        del new_frame, sub_clip, clip
        gc.collect()

# 메모리 삭제
del clip
gc.collect()

process = psutil.Process()
memory_info = process.memory_info()
print("현재 메모리 사용량 (MB):", memory_info.rss / 1024 / 1024)
print(f"===={subtitle['number']} 동영상 완료====")
print("======================================")

del subtitles
gc.collect()

# video에 transition 적용
sub_video = sub_titles_transition(out_path)
sub_video.write_videofile(f"{out_path}/sub_video_example.mp4", fps=60, codec='mpeg4')


head_clip = make_head(img, title, second, size, font) # image에 해당하는 클립 생성
head_clip.write_videofile(f"{out_path}/head_clip.mp4", fps=60, codec='mpeg4')
end_clip = make_end(sub_video, text, second) # 마지막 마무리 영상 생성
end_clip.write_videofile(f"{out_path}/end_clip.mp4", fps=60, codec='mpeg4')

sub_video.audio = audio_file
final_video = concatenate_videoclips([head_clip, sub_video])
final_video = concatenate_videoclips([final_video.set_end(final_video.duration-5), end_clip])
final_video.write_videofile(f"{out_path}/final_shorts_example.mp4", fps=final_video.fps, codec='mpeg4')

