# -*- coding: utf-8 -*-
"""全量训练脚本：fruits-360 全部 131 类蔬果
改进点（提高静态图片与摄像头实时识别准确率）：
1. 使用数据集中的全部图片（训练集 67692 张 / 测试集 22688 张）
2. 每张图片在 30 轮训练中被反复学习（多 Epoch）
3. 在线数据增强：随机翻转 / 旋转 / 缩放 / 亮度 / 对比度 / 饱和度
   —— 模拟真实摄像头的角度与光照差异，提升实时识别泛化能力
4. 改进 CNN：4 个卷积块 + BatchNormalization + 全局平均池化 + Dropout
5. 学习率自动衰减 + 早停 + 标签平滑，抑制过拟合
"""
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                                     Dense, Dropout, BatchNormalization, InputLayer)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA_ROOT = "dataset/fruits-360"
IMG_SIZE = (100, 100)
BATCH = 64
EPOCHS = 30

os.makedirs("outputs", exist_ok=True)
os.makedirs("webapp/model", exist_ok=True)

# 131 类中文名称映射
CN_MAP = {
 "Apple Braeburn":"布里本苹果","Apple Crimson Snow":"深红雪苹果","Apple Golden 1":"金苹果1号",
 "Apple Golden 2":"金苹果2号","Apple Golden 3":"金苹果3号","Apple Granny Smith":"青苹果(格兰尼)",
 "Apple Pink Lady":"粉红佳人苹果","Apple Red 1":"红苹果1号","Apple Red 2":"红苹果2号",
 "Apple Red 3":"红苹果3号","Apple Red Delicious":"红蛇果","Apple Red Yellow 1":"红黄苹果1号",
 "Apple Red Yellow 2":"红黄苹果2号","Apricot":"杏","Avocado":"牛油果","Avocado ripe":"熟牛油果",
 "Banana":"香蕉","Banana Lady Finger":"小米蕉","Banana Red":"红香蕉","Beetroot":"甜菜根",
 "Blueberry":"蓝莓","Cactus fruit":"仙人掌果","Cantaloupe 1":"哈密瓜1号","Cantaloupe 2":"哈密瓜2号",
 "Carambula":"杨桃","Cauliflower":"花椰菜","Cherry 1":"樱桃1号","Cherry 2":"樱桃2号",
 "Cherry Rainier":"雷尼尔樱桃","Cherry Wax Black":"黑蜡樱桃","Cherry Wax Red":"红蜡樱桃",
 "Cherry Wax Yellow":"黄蜡樱桃","Chestnut":"板栗","Clementine":"小柑橘","Cocos":"椰子",
 "Corn":"玉米","Corn Husk":"玉米苞叶","Cucumber Ripe":"黄瓜","Cucumber Ripe 2":"黄瓜2号",
 "Dates":"椰枣","Eggplant":"茄子","Fig":"无花果","Ginger Root":"生姜","Granadilla":"大果西番莲",
 "Grape Blue":"紫葡萄","Grape Pink":"粉红葡萄","Grape White":"白葡萄","Grape White 2":"白葡萄2号",
 "Grape White 3":"白葡萄3号","Grape White 4":"白葡萄4号","Grapefruit Pink":"红心西柚",
 "Grapefruit White":"白西柚","Guava":"番石榴","Hazelnut":"榛子","Huckleberry":"黑果越橘",
 "Kaki":"甜柿","Kiwi":"猕猴桃","Kohlrabi":"苤蓝","Kumquats":"金桔","Lemon":"柠檬",
 "Lemon Meyer":"梅尔柠檬","Limes":"青柠","Lychee":"荔枝","Mandarine":"蜜橘","Mango":"芒果",
 "Mango Red":"红芒果","Mangostan":"山竹","Maracuja":"百香果","Melon Piel de Sapo":"皱纹皮甜瓜",
 "Mulberry":"桑葚","Nectarine":"油桃","Nectarine Flat":"扁平油桃","Nut Forest":"森林坚果",
 "Nut Pecan":"碧根果","Onion Red":"红洋葱","Onion Red Peeled":"去皮红洋葱","Onion White":"白洋葱",
 "Orange":"橙子","Papaya":"木瓜","Passion Fruit":"西番莲果","Peach":"桃子","Peach 2":"桃子2号",
 "Peach Flat":"蟠桃","Pear":"梨","Pear 2":"梨2号","Pear Abate":"阿巴特梨","Pear Forelle":"红彩梨",
 "Pear Kaiser":"凯撒梨","Pear Monster":"丑梨","Pear Red":"红啤梨","Pear Stone":"石梨",
 "Pear Williams":"威廉姆斯梨","Pepino":"人参果","Pepper Green":"青甜椒","Pepper Orange":"橙甜椒",
 "Pepper Red":"红甜椒","Pepper Yellow":"黄甜椒","Physalis":"姑娘果","Physalis with Husk":"带皮姑娘果",
 "Pineapple":"菠萝","Pineapple Mini":"迷你菠萝","Pitahaya Red":"红心火龙果","Plum":"李子",
 "Plum 2":"李子2号","Plum 3":"李子3号","Pomegranate":"石榴","Pomelo Sweetie":"甜心蜜柚",
 "Potato Red":"红皮土豆","Potato Red Washed":"洗净红皮土豆","Potato Sweet":"红薯",
 "Potato White":"白土豆","Quince":"榅桲","Rambutan":"红毛丹","Raspberry":"树莓",
 "Redcurrant":"红醋栗","Salak":"蛇皮果","Strawberry":"草莓","Strawberry Wedge":"切块草莓",
 "Tamarillo":"树番茄","Tangelo":"橘柚","Tomato 1":"番茄1号","Tomato 2":"番茄2号",
 "Tomato 3":"番茄3号","Tomato 4":"番茄4号","Tomato Cherry Red":"圣女果","Tomato Heart":"心形番茄",
 "Tomato Maroon":"褐红番茄","Tomato Yellow":"黄番茄","Tomato not Ripened":"未熟青番茄",
 "Walnut":"核桃","Watermelon":"西瓜",
}
VEG = {"Beetroot","Cauliflower","Corn","Corn Husk","Cucumber Ripe","Cucumber Ripe 2",
       "Eggplant","Ginger Root","Kohlrabi","Onion Red","Onion Red Peeled","Onion White",
       "Pepper Green","Pepper Orange","Pepper Red","Pepper Yellow","Potato Red",
       "Potato Red Washed","Potato Sweet","Potato White","Tomato 1","Tomato 2","Tomato 3",
       "Tomato 4","Tomato Cherry Red","Tomato Heart","Tomato Maroon","Tomato Yellow",
       "Tomato not Ripened"}
NUTS = {"Chestnut","Hazelnut","Nut Forest","Nut Pecan","Walnut"}

# ---------- 1. 全量数据集 ----------
CLASSES = sorted(os.listdir(os.path.join(DATA_ROOT, "Training")))
N = len(CLASSES)
print(f"类别总数: {N}")

def make_dataset(split, augment=False):
    paths, labels = [], []
    for idx, c in enumerate(CLASSES):
        d = os.path.join(DATA_ROOT, split, c)
        for f in os.listdir(d):
            paths.append(os.path.join(d, f))
            labels.append(idx)
    print(f"{split}: {len(paths)} 张")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if augment:
        ds = ds.shuffle(len(paths), seed=42)

    def _load(p, y):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE) / 255.0
        return img, tf.one_hot(y, N)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        rot = tf.keras.layers.RandomRotation(0.15, fill_mode="nearest")
        zoom = tf.keras.layers.RandomZoom(0.15, fill_mode="nearest")

        def _aug(img, y):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.85, 1.15)
            img = tf.image.random_saturation(img, 0.9, 1.1)
            img = rot(img[None])[0]
            img = zoom(img[None])[0]
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, y
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset("Training", augment=True)
test_ds = make_dataset("Test", augment=False)

# ---------- 2. 改进 CNN ----------
def conv_block(filters):
    return [Conv2D(filters, 3, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2, 2)]

model = Sequential(
    [InputLayer(input_shape=(100, 100, 3))]
    + conv_block(32) + conv_block(64) + conv_block(128) + conv_block(256)
    + [GlobalAveragePooling2D(),
       Dense(512, activation="relu"),
       BatchNormalization(),
       Dropout(0.5),
       Dense(N, activation="softmax")])

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=CategoricalCrossentropy(label_smoothing=0.05),
              metrics=["accuracy"])
model.summary()

# ---------- 3. 多轮训练（全部图片 × 30 Epoch） ----------
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                         patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                     restore_best_weights=True, verbose=1),
]
history = model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, callbacks=callbacks)

# ---------- 4. 保存模型与标签 ----------
model.save("webapp/model/fruit_veg_cnn.h5")
with open("webapp/model/labels.json", "w", encoding="utf-8") as f:
    json.dump(CLASSES, f, ensure_ascii=False)
meta = {c: {"cn": CN_MAP.get(c, c),
            "kind": "坚果" if c in NUTS else ("蔬菜" if c in VEG else "水果")}
        for c in CLASSES}
with open("webapp/model/labels_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)

# ---------- 5. 训练曲线 ----------
hist = history.history
ep = range(1, len(hist["accuracy"]) + 1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(ep, hist["accuracy"], label="训练准确率")
plt.plot(ep, hist["val_accuracy"], label="验证准确率")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("准确率曲线（全量131类）")
plt.subplot(1, 2, 2)
plt.plot(ep, hist["loss"], label="训练损失")
plt.plot(ep, hist["val_loss"], label="验证损失")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("损失曲线（全量131类）")
plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=150)
plt.close()

# ---------- 6. 评估 + 混淆矩阵 ----------
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"最终测试集 loss={loss:.4f}, accuracy={acc:.4f}")

y_true, y_pred = [], []
for x, y in test_ds:
    y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1))
    y_true.extend(np.argmax(y.numpy(), axis=1))
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=N).numpy()

# 每类准确率
per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
top_wrong = np.argsort(per_class)[:15]
plt.figure(figsize=(14, 5))
names = [CN_MAP.get(CLASSES[i], CLASSES[i]) for i in top_wrong]
plt.bar(range(len(top_wrong)), per_class[top_wrong], color="#ef5350")
plt.xticks(range(len(top_wrong)), names, rotation=45, ha="right", fontsize=9)
plt.ylabel("召回率"); plt.ylim(0, 1.05); plt.title("识别率最低的15个类别")
plt.tight_layout(); plt.savefig("outputs/per_class_weak.png", dpi=150); plt.close()

plt.figure(figsize=(16, 14))
plt.imshow(cm, cmap="Blues"); plt.colorbar(fraction=0.046)
plt.xticks(range(N), CLASSES, rotation=90, fontsize=5)
plt.yticks(range(N), CLASSES, fontsize=5)
plt.xlabel("预测类别"); plt.ylabel("真实类别"); plt.title(f"混淆矩阵（全量131类，准确率 {acc*100:.2f}%）")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=200); plt.close()

# ---------- 7. 样本识别效果 ----------
show_cls = ["Apple Red 1", "Banana", "Orange", "Strawberry", "Watermelon",
            "Tomato 1", "Cucumber Ripe", "Eggplant", "Pepper Red", "Potato White",
            "Pitahaya Red", "Kiwi"]
fig, axes = plt.subplots(3, 4, figsize=(13, 10))
for ax, c in zip(axes.flat, show_cls):
    fp = os.path.join(DATA_ROOT, "Test", c, sorted(os.listdir(os.path.join(DATA_ROOT, "Test", c)))[0])
    img = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(fp), channels=3), IMG_SIZE) / 255.0
    p = model.predict(img[None], verbose=0)[0]
    j = int(np.argmax(p))
    ax.imshow(img.numpy()); ax.axis("off")
    ok = CLASSES[j] == c
    ax.set_title(f"真:{CN_MAP.get(c,c)}\n预:{CN_MAP.get(CLASSES[j],CLASSES[j])} {p[j]*100:.1f}%",
                 fontsize=9, color="green" if ok else "red")
plt.suptitle("全量模型样本识别效果", fontsize=14)
plt.tight_layout(); plt.savefig("outputs/sample_predictions.png", dpi=150); plt.close()

print("ALL DONE.")
