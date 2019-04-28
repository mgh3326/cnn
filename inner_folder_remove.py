# 모델 로드하기
import os

import cv2

import shutil

JA = ("ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
MO = ("ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ")


def inner_folder_remove(_folder_path):
    for folder_name in os.listdir(_folder_path):
        folder_path = os.path.join(_folder_path, folder_name)

        for file_name in (os.listdir(folder_path)):
            if file_name.split(".")[-1] != "jpg":
                print(file_name)
                shutil.rmtree(os.path.join(folder_path, file_name))


# input_path = "C:/Users/aaa\Downloads\cnn 모델 검증 데이터"
input_path = "C:/tmp\mo_0420_0424"
inner_folder_remove(input_path)
