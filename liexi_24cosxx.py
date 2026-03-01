# -*- coding: utf-8 -*-
# 最终修复版：解决尺寸不均匀 + Sequence初始化警告 + 适配TF 2.16+
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence

# ==================== 1. 全局配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei')

# 核心修改1：指定固定输入尺寸（统一批次形状）
FIXED_INPUT_SIZE = (256, 256)  # 所有图片强制转为256x256，解决形状不均匀

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
BEST_MODEL_PATH = os.path.join(ROOT_DIR, "unet_plus_cos24.h5")
FINAL_MODEL_PATH = os.path.join(ROOT_DIR, "unet_plus_cos_final24.h5")

# 训练参数
BATCH_SIZE = 8
EPOCHS = 30
INIT_LR = 1e-4
MIN_LR = 1e-6
PATIENCE = 10
WEIGHT_DECAY = 1e-5


# ==================== 2. 工具函数 ====================
def cv2_imread_chinese(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"❌ 读取图片失败 {path}：{e}")
        return None


def cv2_imread_mask_chinese(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        mask = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
        return mask
    except Exception as e:
        print(f"❌ 读取掩码失败 {path}：{e}")
        return None


# ==================== 3. 数据生成器（核心修复） ====================
class SegDataGenerator(Sequence):
    """
    修复点：
    1. 构造函数调用 super().__init__(**kwargs) 解决警告
    2. 强制统一尺寸为 FIXED_INPUT_SIZE，解决形状不均匀
    """

    def __init__(self, img_paths, mask_paths, batch_size=8, normalize=True, **kwargs):
        # 核心修复2：调用父类初始化，解决workers参数警告
        super().__init__(**kwargs)

        self.img_paths = shuffle(img_paths)
        self.mask_paths = shuffle(mask_paths)
        self.batch_size = batch_size
        self.normalize = normalize
        self.indexes = np.arange(len(self.img_paths))
        self._filter_invalid_paths()

    def _filter_invalid_paths(self):
        valid_img_paths, valid_mask_paths = [], []
        for img_path, mask_path in zip(self.img_paths, self.mask_paths):
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"⚠️ 文件不存在，跳过：{img_path} / {mask_path}")
                continue
            valid_img_paths.append(img_path)
            valid_mask_paths.append(mask_path)
        self.img_paths = valid_img_paths
        self.mask_paths = valid_mask_paths
        self.indexes = np.arange(len(self.img_paths))
        if len(self.img_paths) == 0:
            raise ValueError("❌ 无有效图片-掩码路径！")
        print(f"✅ 过滤后有效数据量：{len(self.img_paths)}")

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_idx = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_imgs, batch_masks = [], []

        for idx in batch_idx:
            img_path = self.img_paths[idx]
            mask_path = self.mask_paths[idx]

            # 读取图片/掩码
            img = cv2_imread_chinese(img_path)
            if img is None:
                print(f"⚠️ 跳过无效图片：{img_path}")
                continue
            mask = cv2_imread_mask_chinese(mask_path)
            if mask is None:
                print(f"⚠️ 跳过无效掩码：{mask_path}")
                continue

            # 核心修复1：强制统一尺寸 + 补全通道维度
            # 图片强制转为RGB（3通道）
            img = cv2.resize(img, FIXED_INPUT_SIZE, cv2.INTER_LINEAR)
            if len(img.shape) == 2:  # 灰度图转RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA转RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            # 确保最终是 (256,256,3)
            assert img.shape == (*FIXED_INPUT_SIZE, 3), f"图片形状异常：{img.shape}，路径：{img_path}"

            # 掩码强制转为单通道
            mask = cv2.resize(mask, FIXED_INPUT_SIZE, cv2.INTER_NEAREST)
            if len(mask.shape) == 2:  # 确保掩码是2D（防止多通道）
                mask = np.expand_dims(mask, axis=-1)
            # 确保最终是 (256,256,1)
            assert mask.shape == (*FIXED_INPUT_SIZE, 1), f"掩码形状异常：{mask.shape}，路径：{mask_path}"

            # 归一化
            if self.normalize:
                img = img.astype(np.float32) / 255.0
                mask = mask.astype(np.float32) / 255.0

            batch_imgs.append(img)
            batch_masks.append(mask)

        # 核心修复2：兜底处理——确保批次非空且长度等于batch_size
        if len(batch_imgs) == 0:
            raise ValueError(f"❌ 第{index}批次无有效数据！")
        # 补充样本（如果批次样本不足，重复最后一个有效样本）
        while len(batch_imgs) < self.batch_size:
            batch_imgs.append(batch_imgs[-1])
            batch_masks.append(batch_masks[-1])

        # 转为数组（此时所有元素形状完全统一）
        batch_imgs = np.array(batch_imgs, dtype=np.float32)
        batch_masks = np.array(batch_masks, dtype=np.float32)

        # 最终校验
        assert batch_imgs.shape == (self.batch_size, *FIXED_INPUT_SIZE, 3), \
            f"批次图片形状错误：{batch_imgs.shape}，期望：({self.batch_size}, {FIXED_INPUT_SIZE[0]}, {FIXED_INPUT_SIZE[1]}, 3)"
        assert batch_masks.shape == (self.batch_size, *FIXED_INPUT_SIZE, 1), \
            f"批次掩码形状错误：{batch_masks.shape}，期望：({self.batch_size}, {FIXED_INPUT_SIZE[0]}, {FIXED_INPUT_SIZE[1]}, 1)"

        return batch_imgs, batch_masks
    def on_epoch_end(self):
        self.indexes = shuffle(self.indexes)


# ==================== 4. 加载数据集路径 ====================
def load_dataset_paths(data_subdir):
    img_dir = os.path.join(DATASET_DIR, data_subdir, "images")
    mask_dir = os.path.join(DATASET_DIR, data_subdir, "masks")

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        raise ValueError(f"目录不存在：{img_dir} 或 {mask_dir}")

    img_paths, mask_paths = [], []
    img_suffixes = (".jpg", ".jpeg", ".png", ".bmp")

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith(img_suffixes):
            continue

        img_path = os.path.join(img_dir, img_name)
        mask_name = os.path.splitext(img_name)[0]
        mask_path = None
        for suffix in img_suffixes:
            temp_path = os.path.join(mask_dir, mask_name + suffix)
            if os.path.exists(temp_path):
                mask_path = temp_path
                break

        if mask_path:
            img_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            print(f"警告：{img_name} 无对应掩码，跳过")

    if len(img_paths) == 0:
        raise ValueError(f"{data_subdir}目录下无有效图片-掩码对")

    print(f"✅ {data_subdir}集加载完成：{len(img_paths)} 组数据")
    return img_paths, mask_paths


# ==================== 5. 定义U-Net++模型（适配固定尺寸） ====================
def build_unet_plus_plus():
    """核心修改5：模型输入改为固定尺寸 (256,256,3)，避免动态尺寸问题"""
    inputs = Input(shape=(*FIXED_INPUT_SIZE, 3))  # 固定输入尺寸

    # 编码器
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # 瓶颈层
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(c4)

    # 解码器
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u1 = layers.concatenate([u1, c3])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(u1)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(c5)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(u2)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(c6)

    u3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u3 = layers.concatenate([u3, c1])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(u3)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(c7)

    # 输出层
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# ==================== 6. 自定义损失函数 ====================
def dice_loss(y_true, y_pred):
    smooth = 1e-8
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


# ==================== 7. 余弦退火学习率 ====================
def cosine_annealing_lr(epoch, lr):
    T_max = 30
    eta_min = MIN_LR
    new_lr = eta_min + (INIT_LR - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    return new_lr


# ==================== 8. 主训练函数 ====================
def main():
    # 1. 加载数据集
    try:
        train_img_paths, train_mask_paths = load_dataset_paths("train")
        val_img_paths, val_mask_paths = load_dataset_paths("val")
    except Exception as e:
        print(f"❌ 加载数据集失败：{e}")
        return

    # 2. 创建数据生成器
    print("\n📥 创建训练数据生成器...")
    train_generator = SegDataGenerator(
        train_img_paths, train_mask_paths,
        batch_size=BATCH_SIZE, normalize=True
    )
    print("\n📥 创建验证数据生成器...")
    val_generator = SegDataGenerator(
        val_img_paths, val_mask_paths,
        batch_size=BATCH_SIZE, normalize=True
    )

    # 3. 构建模型
    print("\n🚀 构建U-Net++模型...")
    model = build_unet_plus_plus()
    model.summary()

    # 4. 配置优化器
    optimizer = Adam(
        learning_rate=INIT_LR,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        weight_decay=WEIGHT_DECAY
    )

    # 5. 编译模型
    model.compile(
        optimizer=optimizer,
        loss=bce_dice_loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.MeanIoU(num_classes=2, name='miou'),
            dice_loss
        ]
    )

    # 6. 回调函数
    callbacks = [
        LearningRateScheduler(cosine_annealing_lr, verbose=1),
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_miou',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_miou',
            patience=PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger("training_log.csv")
    ]

    # 7. 开始训练（无无效参数）
    print("\n🔥 开始训练...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # 8. 保存模型
    model.save(FINAL_MODEL_PATH)
    print(f"\n✅ 训练完成！")
    print(f"最优模型：{BEST_MODEL_PATH}")
    print(f"最终模型：{FINAL_MODEL_PATH}")

    # 9. 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('训练/验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['miou'], label='训练IoU')
    plt.plot(history.history['val_miou'], label='验证IoU')
    plt.title('训练/验证IoU曲线')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== 9. 程序入口 ====================
if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 显存不足时启用
    main()