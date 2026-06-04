"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import os
import shutil
import json
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from _00_config import Config
"""==============================================================
# 기본설정
=============================================================="""
# 설정값 객체 생성
config = Config()

# 랜덤 시드 설정
rng = np.random.default_rng(seed=config.RANDOM_SEED)

# 클래스 이름 및 라벨 설정
class_name_to_label = config.class_name_to_label
class_names = list(class_name_to_label.keys())

# 이미지 크기 설정
img_height = config.MODEL_IMAGE_HEIGHT
img_width = config.MODEL_IMAGE_WIDTH

# 이미지 확장자 설정
image_extensions = config.IMAGE_EXTENSIONS

# 참조 이미지 폴더 생성 (존재하지 않을 경우에 한함)
os.makedirs(name=config.REFERENCE_BG_DIR, exist_ok=True)
os.makedirs(name=config.REFERENCE_FG_DIR, exist_ok=True)

# 이미지 생성 시, 기존 데이터 삭제 (True로 설정한 경우에 한함)
if config.RECREATE_DATASET is True:
    for target_dir in [config.TRAIN_DIR, config.VALID_DIR, config.TEST_DIR]:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

# 이미지 생성 시, 저장 폴더 생성
os.makedirs(name=config.TRAIN_DIR, exist_ok=True)
os.makedirs(name=config.VALID_DIR, exist_ok=True)
os.makedirs(name=config.TEST_DIR, exist_ok=True)

"""==============================================================
# 데이터 생성
=============================================================="""
# 각 데이터 스플릿별 이미지 개수 카운터
num_train_images = 0
num_valid_images = 0
num_test_images = 0

# 각 데이터 스플릿별 라벨 기록 리스트
train_label_records = []
valid_label_records = []
test_label_records = []

# 참조 이미지 목록 확인
bg_list = sorted(os.listdir(config.REFERENCE_BG_DIR))
fg_class_list = sorted(os.listdir(config.REFERENCE_FG_DIR))

# 이미지 생성 루프
for _ in tqdm(range(config.TRAIN_IMAGE_COUNT + config.VALID_IMAGE_COUNT + config.TEST_IMAGE_COUNT), desc="Synthetic Data generation"):
    # 데이터 스플릿 결정
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

    # 배경 이미지 선택
    bg_selected = rng.choice(bg_list)

    # 전경 이미지 선택
    prob_dist = [config.BG_SCALE_RATIO] + \
        [(1.0 - config.BG_SCALE_RATIO) / (len(fg_class_list) - 1)] * \
        (len(fg_class_list) - 1)
    fg_class_selected = rng.choice(fg_class_list, p=prob_dist)
    label_text, class_name = fg_class_selected.split("_", 1)
    class_label = int(label_text)

    # 이미지 불러오기
    bg = cv2.imread(os.path.join(config.REFERENCE_BG_DIR, bg_selected))

    # 배경 이미지 스케일링
    bg = cv2.resize(src=bg,
                    dsize=(img_width, img_height),
                    interpolation=cv2.INTER_AREA)

    # 배경 어그멘테이션 파라미터
    bg_hue_limit = int(rng.integers(low=2, high=8))
    bg_saturation_fade = int(rng.integers(low=10, high=31))
    bg_saturation_vivid = int(rng.integers(low=10, high=31))
    bg_value_fade = int(rng.integers(low=5, high=16))
    bg_value_vivid = int(rng.integers(low=10, high=26))
    bg_brightness_limit = float(rng.uniform(low=0.03, high=0.12))
    bg_contrast_limit = float(rng.uniform(low=0.03, high=0.15))
    bg_noise_min = float(rng.uniform(low=1.0, high=5.0))
    bg_noise_max = float(rng.uniform(low=8.0, high=30.0))
    bg_shadow_count = int(rng.integers(low=1, high=3))
    bg_shadow_dimension = int(rng.integers(low=2, high=5))
    bg_flare_count = 2
    bg_flare_radius = int(rng.integers(low=15, high=51))

    # 배경 이미지 어그멘테이션
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
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=bg_brightness_limit,
                                   contrast_limit=bg_contrast_limit,
                                   p=0.5),
        A.GaussNoise(var_limit=(bg_noise_min, bg_noise_max), p=0.25),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                       num_shadows_lower=1,
                       num_shadows_upper=bg_shadow_count,
                       shadow_dimension=bg_shadow_dimension,
                       p=0.25),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1),
                         angle_lower=0,
                         angle_upper=1,
                         num_flare_circles_lower=1,
                         num_flare_circles_upper=bg_flare_count,
                         src_radius=bg_flare_radius,
                         p=0.15),
    ])
    bg = bg_transform(image=bg)["image"]


    # 전경 이미지 선택 및 불러오기
    fg_list = sorted(os.listdir(os.path.join(
        config.REFERENCE_FG_DIR, fg_class_selected)))
    fg_selected = rng.choice(fg_list)
    fg = cv2.imread(os.path.join(config.REFERENCE_FG_DIR,
                    fg_class_selected, fg_selected), cv2.IMREAD_UNCHANGED)

    # 전경 이미지 스케일링
    bg_height, bg_width = bg.shape[:2]
    fg_height, fg_width = fg.shape[:2]
    max_scale_width = bg_width / fg_width
    max_scale_height = bg_height / fg_height
    max_scale = min(max_scale_width, max_scale_height)
    random_scale = rng.uniform(low=config.FG_SCALE_RATIO_MIN,
                                high=config.FG_SCALE_RATIO_MAX)
    fg_scale = max_scale * random_scale
    fg_width_scaled = max(int(fg_width * fg_scale), 1)
    fg_height_scaled = max(int(fg_height * fg_scale), 1)
    fg = cv2.resize(src=fg,
                    dsize=(fg_width_scaled, fg_height_scaled),
                    interpolation=cv2.INTER_AREA)

    # 전경 알파 분리
    fg_image = fg[:, :, :3]
    fg_mask = fg[:, :, 3]

    # 전경 어그멘테이션 파라미터
    fg_rotate_limit = int(rng.integers(low=0, high=5))
    fg_perspective_min = float(rng.uniform(low=0.002, high=0.01))
    fg_perspective_max = float(rng.uniform(low=0.015, high=0.04))
    fg_hue_limit = int(rng.integers(low=2, high=8))
    fg_saturation_fade = int(rng.integers(low=10, high=31))
    fg_saturation_vivid = int(rng.integers(low=10, high=31))
    fg_value_fade = int(rng.integers(low=5, high=16))
    fg_value_vivid = int(rng.integers(low=10, high=26))
    fg_brightness_limit = float(rng.uniform(low=0.03, high=0.12))
    fg_contrast_limit = float(rng.uniform(low=0.03, high=0.15))
    fg_noise_min = float(rng.uniform(low=1.0, high=5.0))
    fg_noise_max = float(rng.uniform(low=8.0, high=30.0))
    fg_shadow_count = int(rng.integers(low=1, high=3))
    fg_shadow_dimension = int(rng.integers(low=2, high=5))
    fg_flare_count = 2
    fg_flare_radius = int(rng.integers(low=8, high=31))

    # 전경 이미지 어그멘테이션
    fg_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(rotate=(-fg_rotate_limit, fg_rotate_limit),
                    mode=cv2.BORDER_CONSTANT,
                    cval=(255, 255, 255),
                    cval_mask=0,
                    p=0.8),
        A.Perspective(scale=(fg_perspective_min, fg_perspective_max),
                        pad_mode=cv2.BORDER_CONSTANT,
                        pad_val=(255, 255, 255),
                        mask_pad_val=0,
                        p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=fg_hue_limit,
                                    sat_shift_limit=(-fg_saturation_fade, -10),
                                    val_shift_limit=(-fg_value_fade, 10),
                                    p=1.0),
            A.HueSaturationValue(hue_shift_limit=fg_hue_limit,
                                    sat_shift_limit=(10, fg_saturation_vivid),
                                    val_shift_limit=(10, fg_value_vivid),
                                    p=1.0),
        ], p=0.6),
        A.RandomBrightnessContrast(brightness_limit=fg_brightness_limit,
                                    contrast_limit=fg_contrast_limit,
                                    p=0.6),
        A.GaussNoise(var_limit=(fg_noise_min, fg_noise_max), p=0.25),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                        num_shadows_lower=1,
                        num_shadows_upper=fg_shadow_count,
                        shadow_dimension=fg_shadow_dimension,
                        p=0.25),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1),
                            angle_lower=0,
                            angle_upper=1,
                            num_flare_circles_lower=1,
                            num_flare_circles_upper=fg_flare_count,
                            src_radius=fg_flare_radius,
                            p=0.15),
    ])
    fg_transformed = fg_transform(image=fg_image, mask=fg_mask)
    fg_image = fg_transformed["image"]
    fg_mask = fg_transformed["mask"]
    fg = np.dstack([fg_image, fg_mask])

    # 전경 위치 설정
    fg_height, fg_width = fg.shape[:2]
    margin_x = bg_width - fg_width
    margin_y = bg_height - fg_height
    fg_left = int(rng.integers(low=0, high=margin_x + 1))
    fg_top = int(rng.integers(low=0, high=margin_y + 1))
    fg_right = fg_left + fg_width
    fg_bottom = fg_top + fg_height

    # 이미지 합성
    bg_patch = bg[fg_top:fg_bottom,
                    fg_left:fg_right].astype(np.float32)
    fg_rgb = fg[:, :, :3].astype(np.float32)
    fg_alpha = fg[:, :, 3].astype(np.float32) / 255.0
    fg_alpha = fg_alpha[:, :, None]
    bg_patch = bg_patch * (1.0 - fg_alpha) + fg_rgb * fg_alpha
    bg[fg_top:fg_bottom,
        fg_left:fg_right] = np.clip(bg_patch, 0, 255).astype(np.uint8)

    # 이미지 저장
    file_name = f"{image_idx:06d}" + config.IMAGE_EXTENSIONS
    cv2.imwrite(filename=os.path.join(target_dir, file_name), img=bg)

    # 라벨 기록
    if split_name == "train":
        train_label_records.append({"file_name": file_name,
                                    "label": class_label,
                                    "class_name": class_name,
                                    })
    elif split_name == "valid":
        valid_label_records.append({"file_name": file_name,
                                    "label": class_label,
                                    "class_name": class_name,
                                    })
    elif split_name == "test":
        test_label_records.append({"file_name": file_name,
                                   "label": class_label,
                                   "class_name": class_name,
                                   })
    else:
        raise ValueError("Invalid split name")

"""==============================================================
# 라벨 저장
=============================================================="""
# 라벨 저장
with open(file=config.TRAIN_LABEL_PATH, mode="w", encoding="utf-8") as f:
    json.dump(obj=train_label_records, fp=f, ensure_ascii=False, indent=2)

with open(file=config.VALID_LABEL_PATH, mode="w", encoding="utf-8") as f:
    json.dump(obj=valid_label_records, fp=f, ensure_ascii=False, indent=2)

with open(file=config.TEST_LABEL_PATH, mode="w", encoding="utf-8") as f:
    json.dump(obj=test_label_records, fp=f, ensure_ascii=False, indent=2)
