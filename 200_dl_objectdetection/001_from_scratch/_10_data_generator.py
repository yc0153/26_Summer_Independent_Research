"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import json
import os
import shutil

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

from _00_config import Config

"""==============================================================
# 기본 설정
=============================================================="""
# 설정값 객체 생성
config = Config()

# random seed 설정
rng = np.random.default_rng(seed=config.RANDOM_SEED)

# 클래스 이름과 label mapping 설정
class_name_to_label = config.class_name_to_label

# 모델 입력 이미지 크기 설정
img_height = config.MODEL_IMAGE_HEIGHT
img_width = config.MODEL_IMAGE_WIDTH

# reference image directory 생성
os.makedirs(name=config.REFERENCE_BG_DIR, exist_ok=True)
os.makedirs(name=config.REFERENCE_FG_DIR, exist_ok=True)

# 기존 train/valid/test 데이터를 다시 만들지 여부 확인
if config.RECREATE_DATASET is True:
    for target_dir in [config.TRAIN_DIR, config.VALID_DIR, config.TEST_DIR]:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

# 생성 이미지를 저장할 directory 생성
os.makedirs(name=config.TRAIN_DIR, exist_ok=True)
os.makedirs(name=config.VALID_DIR, exist_ok=True)
os.makedirs(name=config.TEST_DIR, exist_ok=True)

# reference image 목록 확인
bg_list = sorted(os.listdir(config.REFERENCE_BG_DIR))
fg_class_list = sorted(os.listdir(config.REFERENCE_FG_DIR))

"""==============================================================
# 데이터 생성 준비
=============================================================="""
# 각 split에 생성한 이미지 개수 count
num_train_images = 0
num_valid_images = 0
num_test_images = 0

# 각 split에 저장할 label record list
train_label_records = []
valid_label_records = []
test_label_records = []

# train/valid/test 전체 생성 이미지 개수
total_image_count = (
    config.TRAIN_IMAGE_COUNT +
    config.VALID_IMAGE_COUNT +
    config.TEST_IMAGE_COUNT
)

"""==============================================================
# 합성 이미지와 bbox label 생성
=============================================================="""
# train/valid/test 전체 생성 개수만큼 반복
for _ in tqdm(range(total_image_count), desc="Synthetic Data generation"):
    # 현재 생성할 데이터의 split 결정
    if num_train_images < config.TRAIN_IMAGE_COUNT:
        split_name = "train"
        target_dir = config.TRAIN_DIR
        image_idx = num_train_images
        num_train_images += 1
    elif num_valid_images < config.VALID_IMAGE_COUNT:
        split_name = "valid"
        target_dir = config.VALID_DIR
        image_idx = num_valid_images
        num_valid_images += 1
    elif num_test_images < config.TEST_IMAGE_COUNT:
        split_name = "test"
        target_dir = config.TEST_DIR
        image_idx = num_test_images
        num_test_images += 1
    else:
        break

    """==============================================================
    ## 배경 이미지와 전경 클래스 선택
    =============================================================="""
    # 배경 이미지 선택
    bg_selected = rng.choice(bg_list)

    # background class 비율을 반영해서 전경 클래스 선택
    prob_dist = [config.BG_RATIO] + \
        [(1.0 - config.BG_RATIO) / (len(fg_class_list) - 1)] * \
        (len(fg_class_list) - 1)
    fg_class_selected = rng.choice(fg_class_list, p=prob_dist)

    # 폴더명 예시: "1_mouse" -> label 1, class name "mouse"
    label_text, class_name = fg_class_selected.split("_", 1)
    class_label = int(label_text)

    """==============================================================
    ## 배경 이미지 불러오기, 크기 조정, augmentation
    =============================================================="""
    # 배경 이미지 불러오기
    bg_path = os.path.join(config.REFERENCE_BG_DIR, bg_selected)
    bg = cv2.imread(filename=bg_path, flags=cv2.IMREAD_COLOR)

    # 배경 이미지를 모델 입력 크기로 resize
    bg = cv2.resize(src=bg,
                    dsize=(img_width, img_height),
                    interpolation=cv2.INTER_AREA)

    # 배경 augmentation 강도 random 설정
    bg_hue_limit = int(rng.integers(low=4, high=16))
    bg_saturation_fade = int(rng.integers(low=20, high=61))
    bg_saturation_vivid = int(rng.integers(low=20, high=61))
    bg_value_fade = int(rng.integers(low=10, high=31))
    bg_value_vivid = int(rng.integers(low=15, high=41))
    bg_brightness_limit = float(rng.uniform(low=0.08, high=0.22))
    bg_contrast_limit = float(rng.uniform(low=0.08, high=0.25))
    bg_noise_min = float(rng.uniform(low=5.0, high=15.0))
    bg_noise_max = float(rng.uniform(low=25.0, high=70.0))
    bg_shadow_count = int(rng.integers(low=2, high=5))
    bg_shadow_dimension = int(rng.integers(low=4, high=8))
    bg_flare_count = 4
    bg_flare_radius = int(rng.integers(low=30, high=91))

    # 배경 이미지 augmentation 적용
    bg_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=bg_hue_limit,
                                 sat_shift_limit=(-bg_saturation_fade, -10),
                                 val_shift_limit=(-bg_value_fade, 10),
                                 p=1.0),
            A.HueSaturationValue(hue_shift_limit=bg_hue_limit,
                                 sat_shift_limit=(10, bg_saturation_vivid),
                                 val_shift_limit=(10, bg_value_vivid),
                                 p=1.0),
        ], p=0.7),
        A.RandomBrightnessContrast(brightness_limit=bg_brightness_limit,
                                   contrast_limit=bg_contrast_limit,
                                   p=0.7),
        A.GaussNoise(var_limit=(bg_noise_min, bg_noise_max), p=0.4),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                       num_shadows_lower=1,
                       num_shadows_upper=bg_shadow_count,
                       shadow_dimension=bg_shadow_dimension,
                       p=0.35),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1),
                         angle_lower=0,
                         angle_upper=1,
                         num_flare_circles_lower=1,
                         num_flare_circles_upper=bg_flare_count,
                         src_radius=bg_flare_radius,
                         p=0.25),
    ])
    bg = bg_transform(image=bg)["image"]

    # bbox 기본값, 전경의 visible 영역을 찾은 뒤 실제 bbox로 갱신
    bbox = [0.0, 0.0, 0.0, 0.0]

    """==============================================================
    ## 전경 이미지 합성과 bbox 생성
    =============================================================="""
    # 선택된 class의 전경 이미지 중 하나 선택
    fg_dir = os.path.join(config.REFERENCE_FG_DIR, fg_class_selected)
    fg_list = sorted(os.listdir(fg_dir))
    fg_selected = rng.choice(fg_list)

    # 전경 이미지는 alpha channel까지 포함해서 불러오기
    fg_path = os.path.join(fg_dir, fg_selected)
    fg = cv2.imread(filename=fg_path, flags=cv2.IMREAD_UNCHANGED)

    # 좌우 반전 augmentation
    if rng.random() < 0.5:
        fg = cv2.flip(src=fg, flipCode=1)

    # 배경 안에 들어갈 수 있는 최대 크기를 기준으로 random resize
    fg_height, fg_width = fg.shape[:2]
    max_scale_width = img_width / fg_width
    max_scale_height = img_height / fg_height
    max_scale = min(max_scale_width, max_scale_height)
    random_scale = rng.uniform(low=config.FG_SCALE_MIN,
                                high=config.FG_SCALE_MAX)
    fg_scale = max_scale * random_scale

    fg_width_scaled = max(int(fg_width * fg_scale), 1)
    fg_height_scaled = max(int(fg_height * fg_scale), 1)
    fg = cv2.resize(src=fg,
                    dsize=(fg_width_scaled, fg_height_scaled),
                    interpolation=cv2.INTER_AREA)

    # 전경 이미지와 alpha mask 분리
    fg_image = fg[:, :, :3]
    fg_mask = fg[:, :, 3]

    # 전경 augmentation 강도 random 설정
    fg_rotate_limit = int(rng.integers(low=3, high=13))
    fg_perspective_min = float(rng.uniform(low=0.005, high=0.02))
    fg_perspective_max = float(rng.uniform(low=0.03, high=0.07))
    fg_hue_limit = int(rng.integers(low=4, high=13))
    fg_saturation_fade = int(rng.integers(low=15, high=46))
    fg_saturation_vivid = int(rng.integers(low=15, high=46))
    fg_value_fade = int(rng.integers(low=8, high=26))
    fg_value_vivid = int(rng.integers(low=15, high=36))
    fg_brightness_limit = float(rng.uniform(low=0.06, high=0.20))
    fg_contrast_limit = float(rng.uniform(low=0.06, high=0.22))
    fg_noise_min = float(rng.uniform(low=3.0, high=12.0))
    fg_noise_max = float(rng.uniform(low=15.0, high=55.0))
    fg_shadow_count = int(rng.integers(low=1, high=4))
    fg_shadow_dimension = int(rng.integers(low=3, high=7))
    fg_flare_count = 3
    fg_flare_radius = int(rng.integers(low=12, high=46))

    # 전경 이미지와 alpha mask에 같은 geometric transform 적용
    fg_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(rotate=(-fg_rotate_limit, fg_rotate_limit),
                 mode=cv2.BORDER_CONSTANT,
                 cval=(255, 255, 255),
                 cval_mask=0,
                 p=0.9),
        A.Perspective(scale=(fg_perspective_min, fg_perspective_max),
                      pad_mode=cv2.BORDER_CONSTANT,
                      pad_val=(255, 255, 255),
                      mask_pad_val=0,
                      p=0.65),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=fg_hue_limit,
                                 sat_shift_limit=(-fg_saturation_fade, -10),
                                 val_shift_limit=(-fg_value_fade, 10),
                                 p=1.0),
            A.HueSaturationValue(hue_shift_limit=fg_hue_limit,
                                 sat_shift_limit=(10, fg_saturation_vivid),
                                 val_shift_limit=(10, fg_value_vivid),
                                 p=1.0),
        ], p=0.75),
        A.RandomBrightnessContrast(brightness_limit=fg_brightness_limit,
                                   contrast_limit=fg_contrast_limit,
                                   p=0.75),
        A.GaussNoise(var_limit=(fg_noise_min, fg_noise_max), p=0.35),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                       num_shadows_lower=1,
                       num_shadows_upper=fg_shadow_count,
                       shadow_dimension=fg_shadow_dimension,
                       p=0.3),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1),
                         angle_lower=0,
                         angle_upper=1,
                         num_flare_circles_lower=1,
                         num_flare_circles_upper=fg_flare_count,
                         src_radius=fg_flare_radius,
                         p=0.2),
    ])
    fg_transformed = fg_transform(image=fg_image, mask=fg_mask)
    fg_image = fg_transformed["image"]
    fg_mask = fg_transformed["mask"]
    fg = np.dstack([fg_image, fg_mask])

    # alpha가 약한 가장자리 노이즈는 배경으로 정리
    alpha_is_valid = fg[:, :, 3] > config.FG_ALPHA_THRESHOLD
    fg[:, :, 3] = np.where(alpha_is_valid, fg[:, :, 3], 0)

    # 전경 이미지를 배경 위의 random position에 배치
    fg_height, fg_width = fg.shape[:2]
    margin_x = img_width - fg_width
    margin_y = img_height - fg_height
    fg_left = int(rng.integers(low=0, high=margin_x + 1))
    fg_top = int(rng.integers(low=0, high=margin_y + 1))
    fg_right = fg_left + fg_width
    fg_bottom = fg_top + fg_height

    # alpha가 충분히 큰 pixel만 실제 물체 영역으로 사용
    alpha_mask = fg[:, :, 3] > 0

    # background class는 bbox label을 만들지 않고 기본값을 그대로 사용
    if class_label != class_name_to_label["background"]:
        visible_y, visible_x = np.where(alpha_mask)

        if visible_x.size > 0:
            # bbox는 pixel 좌표가 아니라 0~1 사이의 비율 좌표로 저장
            bbox = [
                float((fg_left + visible_x.min()) / img_width),
                float((fg_top + visible_y.min()) / img_height),
                float((fg_left + visible_x.max() + 1) / img_width),
                float((fg_top + visible_y.max() + 1) / img_height),
            ]

    # alpha blending으로 전경을 배경 위에 합성
    bg_patch = bg[fg_top:fg_bottom,
                    fg_left:fg_right].astype(np.float32)
    fg_rgb = fg[:, :, :3].astype(np.float32)
    fg_alpha = fg[:, :, 3].astype(np.float32) / 255.0
    fg_alpha = fg_alpha[:, :, None]

    bg_patch = bg_patch * (1.0 - fg_alpha) + fg_rgb * fg_alpha
    bg[fg_top:fg_bottom,
        fg_left:fg_right] = np.clip(bg_patch, 0, 255).astype(np.uint8)

    """==============================================================
    ## 이미지와 label record 저장
    =============================================================="""
    # 합성 이미지 저장
    file_name = f"{image_idx:06d}" + config.IMAGE_EXTENSIONS
    cv2.imwrite(filename=os.path.join(target_dir, file_name), img=bg)

    # label json에 저장할 record 생성
    label_record = {
        "file_name": file_name,
        "label": class_label,
        "class_name": class_name,
        "bbox": bbox,
    }

    # 현재 split에 맞는 label record list에 추가
    if split_name == "train":
        train_label_records.append(label_record)
    elif split_name == "valid":
        valid_label_records.append(label_record)
    elif split_name == "test":
        test_label_records.append(label_record)
    else:
        raise ValueError("Invalid split name")

"""==============================================================
# label 파일 저장
=============================================================="""
# train/valid/test label record를 json file로 저장
with open(file=config.TRAIN_LABEL_PATH, mode="w", encoding="utf-8") as f:
    json.dump(obj=train_label_records, fp=f, ensure_ascii=False, indent=2)

with open(file=config.VALID_LABEL_PATH, mode="w", encoding="utf-8") as f:
    json.dump(obj=valid_label_records, fp=f, ensure_ascii=False, indent=2)

with open(file=config.TEST_LABEL_PATH, mode="w", encoding="utf-8") as f:
    json.dump(obj=test_label_records, fp=f, ensure_ascii=False, indent=2)
