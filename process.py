# # Change root paths to your actual directories
# image_dirs = [
#     r"D:\bharat\FAFuse-master\data\ISIC2018_Task1-2_Training_Input",
#     r"D:\bharat\FAFuse-master\data\ISIC2018_Task1-2_Validation_Input"
#     r"D:\bharat\FAFuse-master\data\ISIC2018_Task1-2_Test_Input"
# ]
# mask_dirs = [
#     r"D:\bharat\FAFuse-master\data\ISIC2018_Task1_Training_GroundTruth",
#     r"D:\bharat\FAFuse-master\data\ISIC2018_Task1_Validation_GroundTruth"
#     r"D:\bharat\FAFuse-master\data\ISIC2018_Task1_Test_GroundTruth"
# ]
import numpy as np
import cv2
import os

image_dirs = [
    r"D:\bharat\FAFuse-master\data\ISIC2018_Task1-2_Training_Input",
    r"D:\bharat\FAFuse-master\data\ISIC2018_Task1-2_Validation_Input",
    r"D:\bharat\FAFuse-master\data\ISIC2018_Task1-2_Test_Input"
]
mask_dirs = [
    r"D:\bharat\FAFuse-master\data\ISIC2018_Task1_Training_GroundTruth",
    r"D:\bharat\FAFuse-master\data\ISIC2018_Task1_Validation_GroundTruth",
    r"D:\bharat\FAFuse-master\data\ISIC2018_Task1_Test_GroundTruth"
]

save_name = ['train', 'test','Validation']
height, width = 352, 352

for j in range(len(image_dirs)):
    print(f"Processing {image_dirs[j]}...")

    imgs_list = []
    masks_list = []

    img_files = sorted(os.listdir(image_dirs[j]))

    for fname in img_files:
        img_path = os.path.join(image_dirs[j], fname)
        name, _ = os.path.splitext(fname)
        mask_name = name + "_segmentation.png"
        mask_path = os.path.join(mask_dirs[j], mask_name)

        if not os.path.exists(img_path):
            print("❌ Missing image:", img_path)
            continue
        if not os.path.exists(mask_path):
            print("❌ Missing mask:", mask_path)
            continue

        img = cv2.imread(img_path)
        if img is None:
            print("⚠️ Failed to read:", img_path)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("⚠️ Failed to read mask:", mask_path)
            continue
        mask = cv2.resize(mask, (width, height))

        imgs_list.append(img)
        masks_list.append(mask)

        print(f"[{len(imgs_list)}/{len(img_files)}] {fname}")

    imgs = np.array(imgs_list, dtype=np.uint8)
    masks = np.array(masks_list, dtype=np.uint8)

    np.save(f"{os.path.dirname(image_dirs[j])}/data_{save_name[j]}.npy", imgs)
    np.save(f"{os.path.dirname(mask_dirs[j])}/mask_{save_name[j]}.npy", masks)

print("✅ All datasets processed!")

