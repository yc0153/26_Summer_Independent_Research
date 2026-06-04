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
# 기본 설정
=============================================================="""
# 설정값 객체 생성
config = Config()

# random seed 설정
rng = np.random.default_rng(seed=config.RANDOM_SEED)

# 클래스 이름과 라벨 설정
class_name_to_label = config.class_name_to_label
class_names = list(class_name_to_label.keys())

# 모델 입력 이미지 크기 설정
img_height = config.MODEL_IMAGE_HEIGHT
img_width = config.MODEL_IMAGE_WIDTH

# 생성 이미지 확장자 설정
image_extensions = config.IMAGE_EXTENSIONS

# reference image directory 생성
os.makedirs(name=config.REFERENCE_BG_DIR, exist_ok=True)
os.makedirs(name=config.REFERENCE_FG_DIR, exist_ok=True)

# 기존 train/valid/test 데이터를 다시 만들지 여부 확인
if config.RECREATE_DATASET is True:
    for target_dir in [config.TRAIN_DIR, config.VALID_DIR, config.TEST_DIR]:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

# 생성 이미지 저장 directory 생성
os.makedirs(name=config.TRAIN_DIR, exist_ok=True)
os.makedirs(name=config.VALID_DIR, exist_ok=True)
os.makedirs(name=config.TEST_DIR, exist_ok=True)

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

"""==============================================================
# 합성 이미지와 라벨 생성
=============================================================="""
# train/valid/test 전체 생성 개수만큼 반복
for _ in tqdm(range(config.TRAIN_IMAGE_COUNT + config.VALID_IMAGE_COUNT + config.TEST_IMAGE_COUNT), desc="Synthetic Data generation"):
    # 현재 생성할 데이터 split 결정
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
    ## 배경 이미지와 전경 이미지 선택
    =============================================================="""
    # 배경 이미지 선택
    bg_list = os.listdir(config.REFERENCE_BG_DIR)
    bg_selected = np.random.choice(bg_list)

    # 전경 클래스와 전경 이미지 선택
    fg_class_list = os.listdir(config.REFERENCE_FG_DIR)
    fg_class_selected = np.random.choice(fg_class_list)
    fg_list = os.listdir(os.path.join(
        config.REFERENCE_FG_DIR, fg_class_selected))
    fg_selected = np.random.choice(fg_list)

    # 이미지 불러오기
    bg = cv2.imread(os.path.join(config.REFERENCE_BG_DIR, bg_selected))
    fg = cv2.imread(os.path.join(config.REFERENCE_FG_DIR,
                    fg_class_selected, fg_selected), cv2.IMREAD_UNCHANGED)

    """==============================================================
    ## 배경과 전경 크기 조정
    =============================================================="""
    # 배경 이미지를 모델 입력 크기로 resize
    bg = cv2.resize(src=bg,
                    dsize=(img_width, img_height),
                    interpolation=cv2.INTER_AREA)

    # 전경 이미지를 배경 안에 들어갈 수 있는 크기로 random resize
    bg_height, bg_width = bg.shape[:2]
    fg_height, fg_width = fg.shape[:2]
    max_scale_width = bg_width / fg_width
    max_scale_height = bg_height / fg_height
    max_scale = min(max_scale_width, max_scale_height)
    random_scale = rng.uniform(low=0.2, high=1.0)
    fg_scale = max_scale * random_scale
    fg_width_scaled = max(int(fg_width * fg_scale), 1)
    fg_height_scaled = max(int(fg_height * fg_scale), 1)
    fg = cv2.resize(src=fg,
                    dsize=(fg_width_scaled, fg_height_scaled),
                    interpolation=cv2.INTER_AREA)

    """==============================================================
    ## 배경 이미지 augmentation
    =============================================================="""
    # 배경 augmentation 강도 random 설정
    bg_hue_limit = int(rng.integers(low=5, high=16))
    bg_saturation_fade = int(rng.integers(low=20, high=71))
    bg_saturation_vivid = int(rng.integers(low=20, high=71))
    bg_value_fade = int(rng.integers(low=10, high=31))
    bg_value_vivid = int(rng.integers(low=10, high=51))
    bg_brightness_limit = float(rng.uniform(low=0.10, high=0.30))
    bg_contrast_limit = float(rng.uniform(low=0.10, high=0.35))
    bg_noise_min = float(rng.uniform(low=5.0, high=20.0))
    bg_noise_max = float(rng.uniform(low=40.0, high=120.0))
    bg_shadow_count = int(rng.integers(low=1, high=4))
    bg_shadow_dimension = int(rng.integers(low=4, high=8))
    bg_flare_count = int(rng.integers(low=2, high=5))
    bg_flare_radius = int(rng.integers(low=40, high=101))

    # 배경 이미지 augmentation 적용
    bg_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
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
        A.GaussNoise(var_limit=(bg_noise_min, bg_noise_max), p=0.5),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                       num_shadows_lower=1,
                       num_shadows_upper=bg_shadow_count,
                       shadow_dimension=bg_shadow_dimension,
                       p=0.5),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1),
                         angle_lower=0,
                         angle_upper=1,
                         num_flare_circles_lower=1,
                         num_flare_circles_upper=bg_flare_count,
                         src_radius=bg_flare_radius,
                         p=0.4),
    ])
    bg = bg_transform(image=bg)["image"]

    """==============================================================
    ## 전경 이미지 augmentation
    =============================================================="""
    # 전경 이미지와 alpha mask 분리
    fg_image = fg[:, :, :3]
    fg_mask = fg[:, :, 3]

    # 전경 augmentation 강도 random 설정
    fg_rotate_limit = int(rng.integers(low=0, high=5))
    fg_perspective_min = float(rng.uniform(low=0.01, high=0.04))
    fg_perspective_max = float(rng.uniform(low=0.05, high=0.12))
    fg_hue_limit = int(rng.integers(low=5, high=16))
    fg_saturation_fade = int(rng.integers(low=20, high=71))
    fg_saturation_vivid = int(rng.integers(low=20, high=71))
    fg_value_fade = int(rng.integers(low=10, high=31))
    fg_value_vivid = int(rng.integers(low=10, high=51))
    fg_brightness_limit = float(rng.uniform(low=0.10, high=0.30))
    fg_contrast_limit = float(rng.uniform(low=0.10, high=0.35))
    fg_noise_min = float(rng.uniform(low=5.0, high=20.0))
    fg_noise_max = float(rng.uniform(low=40.0, high=120.0))
    fg_shadow_count = int(rng.integers(low=1, high=4))
    fg_shadow_dimension = int(rng.integers(low=4, high=8))
    fg_flare_count = int(rng.integers(low=2, high=5))
    fg_flare_radius = int(rng.integers(low=20, high=81))

    # 전경 이미지와 mask에 같은 geometric transform 적용
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
                      p=0.8),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=fg_hue_limit,
                                 sat_shift_limit=(-fg_saturation_fade, -10),
                                 val_shift_limit=(-fg_value_fade, 10),
                                 p=1.0),
            A.HueSaturationValue(hue_shift_limit=fg_hue_limit,
                                 sat_shift_limit=(10, fg_saturation_vivid),
                                 val_shift_limit=(10, fg_value_vivid),
                                 p=1.0),
        ], p=0.8),
        A.RandomBrightnessContrast(brightness_limit=fg_brightness_limit,
                                   contrast_limit=fg_contrast_limit,
                                   p=0.8),
        A.GaussNoise(var_limit=(fg_noise_min, fg_noise_max), p=0.5),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                       num_shadows_lower=1,
                       num_shadows_upper=fg_shadow_count,
                       shadow_dimension=fg_shadow_dimension,
                       p=0.5),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1),
                         angle_lower=0,
                         angle_upper=1,
                         num_flare_circles_lower=1,
                         num_flare_circles_upper=fg_flare_count,
                         src_radius=fg_flare_radius,
                         p=0.4),
    ])
    fg_transformed = fg_transform(image=fg_image, mask=fg_mask)
    fg_image = fg_transformed["image"]
    fg_mask = fg_transformed["mask"]
    fg = np.dstack([fg_image, fg_mask])

    """==============================================================
    ## 전경 위치와 바운딩 박스 계산
    =============================================================="""
    # 전경을 배경 안쪽의 random position에 배치
    fg_height, fg_width = fg.shape[:2]
    margin_x = bg_width - fg_width
    margin_y = bg_height - fg_height
    fg_left = int(rng.integers(low=0, high=margin_x + 1))
    fg_top = int(rng.integers(low=0, high=margin_y + 1))
    fg_right = fg_left + fg_width
    fg_bottom = fg_top + fg_height

    # alpha mask에서 실제 보이는 영역만 바운딩 박스로 사용
    fg_visible_y, fg_visible_x = np.where(fg[:, :, 3] > 0)
    if len(fg_visible_x) > 0 and len(fg_visible_y) > 0:
        bbox_x_min = fg_left + int(fg_visible_x.min())
        bbox_y_min = fg_top + int(fg_visible_y.min())
        bbox_x_max = fg_left + int(fg_visible_x.max()) + 1
        bbox_y_max = fg_top + int(fg_visible_y.max()) + 1
    else:
        bbox_x_min = 0
        bbox_y_min = 0
        bbox_x_max = 0
        bbox_y_max = 0

    """==============================================================
    ## 배경과 전경 합성
    =============================================================="""
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
    ## 이미지와 라벨 저장
    =============================================================="""
    # 합성 이미지 저장
    file_name = f"{image_idx:06d}" + config.IMAGE_EXTENSIONS
    cv2.imwrite(filename=os.path.join(target_dir, file_name), img=bg)

    # class label과 bbox label record 생성
    label_text, class_name = fg_class_selected.split("_", 1)
    if int(label_text) == 0:
        bbox = {
            "x_min": 0,
            "y_min": 0,
            "x_max": img_width,
            "y_max": img_height,
        }
    else:
        bbox = {
            "x_min": bbox_x_min,
            "y_min": bbox_y_min,
            "x_max": bbox_x_max,
            "y_max": bbox_y_max,
        }

    # 현재 split에 맞는 label record list에 추가
    if split_name == "train":
        train_label_records.append({"file_name": file_name,
                                    "label": int(label_text),
                                    "class_name": class_name,
                                    "bbox": bbox,
                                    })
    elif split_name == "valid":
        valid_label_records.append({"file_name": file_name,
                                    "label": int(label_text),
                                    "class_name": class_name,
                                    "bbox": bbox,
                                    })
    elif split_name == "test":
        test_label_records.append({"file_name": file_name,
                                   "label": int(label_text),
                                   "class_name": class_name,
                                   "bbox": bbox,
                                   })
    else:
        raise ValueError("Invalid split name")

"""==============================================================
# 라벨 파일 저장
=============================================================="""
# train/valid/test label record를 json file로 저장
with open(file=config.TRAIN_LABEL_PATH, mode="w", encoding="utf-8") as f:
    json.dump(obj=train_label_records, fp=f, ensure_ascii=False, indent=2)

with open(file=config.VALID_LABEL_PATH, mode="w", encoding="utf-8") as f:
    json.dump(obj=valid_label_records, fp=f, ensure_ascii=False, indent=2)

with open(file=config.TEST_LABEL_PATH, mode="w", encoding="utf-8") as f:
    json.dump(obj=test_label_records, fp=f, ensure_ascii=False, indent=2)
