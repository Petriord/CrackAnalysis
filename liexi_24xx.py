# -*- coding: utf-8 -*-
# U-Net++ 简化训练文件 - 支持任意尺寸图片训练（修复尺寸不均+警告问题）
import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
import tensorflow as tf


# ==================== 1. 动态尺寸U-Net++模型 ====================
def simple_unet(input_size=(None, None, 3)):
    """修改为支持动态尺寸输入（高/宽需为8的倍数）"""
    inputs = Input(input_size)

    # 编码器
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)  # 尺寸/2

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)  # 尺寸/4

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)  # 尺寸/8

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    # 解码器
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)  # 尺寸*2
    u1 = layers.concatenate([u1, c3])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)  # 尺寸*4
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)  # 尺寸*8
    u3 = layers.concatenate([u3, c1])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u3)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# ==================== 2. 数据生成器（修复尺寸不均+警告问题） ====================
class DataGenerator(Sequence):
    """
    数据生成器：解决不同尺寸图片无法放入同一NumPy数组的问题
    核心修复：强制所有图片使用统一固定尺寸
    """

    def __init__(self, img_paths, mask_paths, batch_size=8, img_size=(512, 512), normalize=True):
        # 修复警告：调用父类构造函数
        super().__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        # 强制使用固定尺寸（必须是8的倍数）
        self.img_size = self._adjust_to_8x(*img_size)
        self.normalize = normalize
        self.indexes = np.arange(len(self.img_paths))

    def __len__(self):
        # 计算每个epoch的批次数
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, index):
        # 生成单个批次数据（保证所有批次尺寸一致）
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_img_paths = [self.img_paths[i] for i in batch_indexes]
        batch_mask_paths = [self.mask_paths[i] for i in batch_indexes]

        # 加载单批次的图片和掩码（强制统一尺寸）
        batch_imgs, batch_masks = self._load_batch(batch_img_paths, batch_mask_paths)

        # 转换为numpy数组（所有批次尺寸一致）
        batch_imgs = np.array(batch_imgs)
        batch_masks = np.array(batch_masks)

        return batch_imgs, batch_masks

    def on_epoch_end(self):
        # 每个epoch后打乱索引
        np.random.shuffle(self.indexes)

    def _adjust_to_8x(self, h, w):
        """将尺寸调整为8的倍数（模型要求）"""
        new_h = (h // 8) * 8 if h % 8 != 0 else h
        new_w = (w // 8) * 8 if w % 8 != 0 else w
        return new_h, new_w

    def _load_batch(self, img_paths, mask_paths):
        imgs = []
        masks = []

        for img_path, mask_path in zip(img_paths, mask_paths):
            # 加载图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取图像 {img_path}")
                continue

            # 加载掩码（灰度图）
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告：无法读取掩码 {mask_path}")
                continue

            # 强制resize到固定尺寸（所有图片统一）
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]),
                             interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]),
                              interpolation=cv2.INTER_NEAREST)  # 掩码用最近邻插值

            # 归一化
            if self.normalize:
                img = img.astype(np.float32) / 255.0
                mask = mask.astype(np.float32) / 255.0

            # 扩展掩码维度 (H,W) -> (H,W,1)
            mask = np.expand_dims(mask, axis=-1)

            imgs.append(img)
            masks.append(mask)

        # 处理空数据情况
        if len(imgs) == 0:
            raise ValueError(f"批次 {index} 未加载到任何有效数据")

        return imgs, masks


# ==================== 3. 路径整理函数 ====================
def get_img_mask_paths(data_dir):
    """整理图片和掩码的路径（匹配文件名）"""
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    # 检查目录是否存在
    if not os.path.exists(img_dir):
        raise ValueError(f"图像目录不存在: {img_dir}")
    if not os.path.exists(mask_dir):
        raise ValueError(f"掩码目录不存在: {mask_dir}")

    img_paths = []
    mask_paths = []

    # 获取所有图片文件
    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"从 {data_dir} 发现 {len(img_files)} 张图像...")

    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        # 匹配掩码（优先同文件名，不同扩展名也尝试）
        mask_base = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            temp_path = os.path.join(mask_dir, mask_base + ext)
            if os.path.exists(temp_path):
                mask_path = temp_path
                break

        if mask_path is not None:
            img_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            print(f"警告：未找到 {img_name} 对应的掩码文件")

    if not img_paths:
        raise ValueError(f"在 {data_dir} 中没有加载到有效图像-掩码对")

    print(f"成功匹配 {len(img_paths)} 组图像-掩码对")
    return img_paths, mask_paths


# ==================== 4. 主训练函数 ====================
def main():
    # 配置参数
    BATCH_SIZE = 8  # 固定尺寸后，根据显存调整批次（512尺寸建议4/2）
    EPOCHS = 30
    INITIAL_LR = 1e-4
    DATASET_ROOT = "dataset"
    TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
    VAL_DIR = os.path.join(DATASET_ROOT, "val")

    # 关键修改：使用固定尺寸（必须是8的倍数）
    # 可选值：(256,256) / (512,512) / (640,640)（根据显存调整）
    IMG_SIZE = (256, 256)

    # 模型保存路径
    BEST_MODEL_SAVE_PATH = "unet_crack_24.h5"
    FINAL_MODEL_SAVE_PATH = "unet_crack_final24.h5"

    # 检查数据集路径
    try:
        train_img_paths, train_mask_paths = get_img_mask_paths(TRAIN_DIR)
        val_img_paths, val_mask_paths = get_img_mask_paths(VAL_DIR)
    except ValueError as e:
        print(f"数据加载失败: {e}")
        print("请确保数据集结构为：")
        print("dataset/")
        print("├── train/")
        print("│   ├── images/")
        print("│   └── masks/")
        print("└── val/")
        print("    ├── images/")
        print("    └── masks/")
        return

    try:
        # 1. 创建数据生成器（强制统一固定尺寸）
        train_generator = DataGenerator(
            train_img_paths, train_mask_paths,
            batch_size=BATCH_SIZE, img_size=IMG_SIZE
        )
        val_generator = DataGenerator(
            val_img_paths, val_mask_paths,
            batch_size=BATCH_SIZE, img_size=IMG_SIZE
        )

        # 2. 创建动态尺寸模型（仍支持任意尺寸，但训练时用固定尺寸）
        print("创建动态尺寸U-Net++模型...")
        model = simple_unet(input_size=(None, None, 3))

        # 3. 编译模型
        model.compile(
            optimizer=Adam(learning_rate=INITIAL_LR),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # 打印模型摘要
        model.summary()

        # 4. 回调函数
        callbacks = [
            ModelCheckpoint(
                BEST_MODEL_SAVE_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
        ]

        # 5. 开始训练（移除workers/use_multiprocessing，修复尺寸问题）
        print("开始训练...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        # 6. 保存最终模型
        model.save(FINAL_MODEL_SAVE_PATH)

        print("=" * 30)
        print("训练完成！")
        print(f"最佳模型: {BEST_MODEL_SAVE_PATH}")
        print(f"最终模型: {FINAL_MODEL_SAVE_PATH}")
        print("=" * 30)

    except Exception as e:
        print(f"训练出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 关键修改：关闭Eager执行（避免形状锁定）
    # tf.config.run_functions_eagerly(True)  # 注释掉这行

    # 限制显存增长（避免大图片训练时显存溢出）
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()