import pandas as pd
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os

# Inisialisasi augmenter
aug = iaa.Sequential([
    iaa.MultiplySaturation((0.6, 1.6)),
    iaa.MultiplySaturation((0.4, 1.4)),
    iaa.MultiplyBrightness((0.65, 1.35)),
    iaa.MultiplyBrightness((0.65, 1.35)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.15*255)),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.AllChannelsHistogramEqualization()
])

# Create Output folder
os.mkdir("output")
os.mkdir("output/train")
os.mkdir("output/valid")
os.mkdir("output/test")
os.mkdir("trash")

# Target csv
folder_target = "target"
csv_train_target = f"{folder_target}/train/train.csv"
csv_valid_target = f"{folder_target}/valid/valid.csv"
csv_test_target = f"{folder_target}/test/test.csv"

# Create removeDupicate function
def remove_duplicates(input_file, output_file):
    # Membaca file CSV menjadi DataFrame
    df = pd.read_csv(input_file)

    # Menghapus record duplicate berdasarkan semua kolom
    df.drop_duplicates(inplace=True)

    # Menyimpan DataFrame yang telah diubah ke file CSV baru
    df.to_csv(output_file, index=False)

    print("Record duplicate telah dihapus. Hasil disimpan ke", output_file)

def augment_start (csv, type) :
    # Membaca data dari file CSV
    df = pd.read_csv(csv)

    # Inisialisasi DataFrame baru untuk hasil augmentasi
    augmented_data = pd.DataFrame(columns=df.columns)

    # Loop melalui setiap baris dalam datafr{ame
    for index, row in df.iterrows():
        # Membaca gambar original
        image_path = f"{folder_target}/{type}/" + row['filename']
        image = cv2.imread(image_path)

        # Membaca anotasi original
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        bounding_box = BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
        bounding_boxes = BoundingBoxesOnImage([bounding_box], shape=image.shape)

        # Simpan gambar original dan anotasinya
        original_image_path = f"output/{type}/{row['filename']}"
        cv2.imwrite(original_image_path, image)

        # Tambahkan baris anotasi untuk gambar original ke DataFrame baru
        original_row = row.copy()
        original_row['filename'] = row['filename']
        augmented_data = pd.concat([augmented_data, pd.DataFrame([original_row])], ignore_index=True)

        # Proses augmentasi untuk setiap gambar hasil augmentasi
        for i in range(7):  # 7 hasil augmentasi
            augmented_image, augmented_bounding_boxes = aug(image=image, bounding_boxes=bounding_boxes)

            # Simpan gambar hasil augmentasi
            augmented_image_path = f"output/{type}/aug_{i}_{row['filename']}"
            cv2.imwrite(augmented_image_path, augmented_image)

            # Duplikat anotasi ke DataFrame untuk gambar hasil augmentasi
            augmented_row = row.copy()
            augmented_row['filename'] = f"aug_{i}_{row['filename']}"
            augmented_row['xmin'] = augmented_bounding_boxes[0].x1
            augmented_row['ymin'] = augmented_bounding_boxes[0].y1
            augmented_row['xmax'] = augmented_bounding_boxes[0].x2
            augmented_row['ymax'] = augmented_bounding_boxes[0].y2

            # Tambahkan baris hasil augmentasi ke DataFrame baru
            augmented_data = pd.concat([augmented_data, pd.DataFrame([augmented_row])], ignore_index=True)

    # Gabungkan DataFrame original dan DataFrame hasil augmentasi
    final_df = pd.concat([df, augmented_data], ignore_index=True)

    # Simpan dataframe baru yang mencakup gambar asli dan hasil augmentasi
    final_df.to_csv(f"trash/{type}_labels.csv", index=False)

    # Menghapus record duplicate
    remove_duplicates(f"trash/{type}_labels.csv", f"output/{type}_labels.csv")


# Jalankan script
augment_start(csv_train_target,"train")
augment_start(csv_valid_target,"valid")
augment_start(csv_test_target,"test")