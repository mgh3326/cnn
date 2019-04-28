# 모델 로드하기
import os

import cv2

import shutil

JA = ("ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
MO = ("ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ")


def evaluate(_folder_path):
    _count = 0
    for folder_name in os.listdir(_folder_path):
        folder_path = os.path.join(_folder_path, folder_name)
        try:
            print(folder_name, len(os.listdir(folder_path)))
        except NotADirectoryError:
            continue
    # for file_name in os.listdir(folder_path):
    #     if file_name.split(".")[-1] != "jpg":
    #         continue
    # return _count


def folder_num_minimum(_folder_path):
    folder_num = []
    for folder_name in os.listdir(_folder_path):
        folder_path = os.path.join(_folder_path, folder_name)
        try:
            folder_num.append(len(os.listdir(folder_path)))
        except NotADirectoryError:
            continue
    return (min(folder_num))


def folder_num_minimum_delete(_folder_path, minimum):
    for folder_name in os.listdir(_folder_path):
        folder_path = os.path.join(_folder_path, folder_name)
        complete_data_path = os.path.join(folder_path, "remove_for_min")
        if not os.path.exists(complete_data_path):
            try:
                os.makedirs(complete_data_path)
            except NotADirectoryError:
                continue
            except FileNotFoundError:
                continue
        for idx, file_name in enumerate(os.listdir(folder_path)):
            if idx > minimum:
                if file_name.split(".")[-1] != "jpg":
                    continue

                shutil.move(os.path.join(folder_path, file_name), os.path.join(complete_data_path, file_name))


# input_path = "C:/Users/aaa\Downloads\cnn 모델 검증 데이터"
input_path = "C:/tmp/mo_0420_0424"
minmun = folder_num_minimum(input_path)
folder_num_minimum_delete(input_path, minmun)
