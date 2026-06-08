# Đồ án Phân loại Nhóm Tuổi qua Khuôn Mặt (Age Group Classification)

Dự án ứng dụng Deep Learning để phân loại nhóm tuổi của người dùng qua khuôn mặt từ ảnh tĩnh hoặc trực tiếp từ luồng Webcam. Dự án đã được chuyển đổi toàn diện từ mô hình hồi quy (dự đoán số tuổi cụ thể) sang mô hình **phân loại nhóm tuổi** (Classification) sử dụng framework **PyTorch** để tăng độ chính xác và ổn định.

---

## Kiến trúc Hệ thống

- **Tập dữ liệu (Dataset)**: UTKFace dataset.
- **Framework**: PyTorch & Torchvision.
- **Mô hình (Model)**: Transfer Learning với kiến trúc **ResNet50** (đóng băng các lớp convolution đầu, mở khóa tinh chỉnh lớp `layer3`, `layer4` và định nghĩa lại lớp Fully Connected cuối cùng).
- **Bộ tiền xử lý**: OpenCV Haar Cascade dùng để tự động nhận diện và cắt khuôn mặt chuẩn hóa (224x224 px) từ ảnh gốc trước khi đưa vào mô hình.
- **Loss Function**: Cross Entropy Loss với Label Smoothing (0.1) để giảm overfitting.
- **Optimizer**: AdamW kết hợp với Cosine Annealing Learning Rate Scheduler.

---

## Cấu trúc Thư mục

```text
📦 deeplearning
 ┣ 📂 .agents/rules    # Quy tắc cấu hình tác vụ cho AI
 ┣ 📂 dataset
 ┃ ┣ 📂 UTKFace        # Dữ liệu ảnh UTKFace gốc (tải thủ công từ Kaggle)
 ┃ ┗ 📂 UTKFace_Cleaned# Ảnh khuôn mặt đã được tiền xử lý và cắt tự động
 ┣ 📂 models           # Nơi lưu trữ checkpoint mô hình tốt nhất (.pth)
 ┣ 📂 results          # Chứa các biểu đồ kết quả sau huấn luyện và đánh giá
 ┃ ┣ 📜 augmented_samples.png
 ┃ ┣ 📜 demo_augmentation.png
 ┃ ┣ 📜 training_report.png
 ┃ ┣ 📜 evaluate_accuracy_per_group.png
 ┃ ┣ 📜 evaluate_confusion_matrix.png
 ┃ ┗ 📜 evaluate_roc_curve.png
 ┣ 📜 requirements.txt # Danh sách các thư viện cần thiết
 ┣ 📜 README.md        # Tài liệu hướng dẫn dự án
 ┣ 📜 Phanloaituoi.ipynb# File Notebook chạy thử nghiệm trên Colab
 ┗ 📂 src
   ┣ 📜 config.py      # Cấu hình tham số huấn luyện và định nghĩa nhóm tuổi
   ┣ 📜 dataset_loader.py # Bộ lọc mặt, PyTorch Dataset, Balanced Sampler và Augmentations
   ┣ 📜 model.py       # Định nghĩa mô hình ResNet50 điều chỉnh lớp phân loại cuối
   ┣ 📜 train.py       # Kịch bản huấn luyện mô hình kết hợp Early Stopping
   ┣ 📜 evaluate.py    # Đánh giá mô hình trên tập kiểm thử (accuracy, confusion matrix, ROC)
   ┣ 📜 predict.py     # Dự đoán nhóm tuổi của một ảnh tĩnh cụ thể
   ┗ 📜 cam.py         # Nhận diện nhóm tuổi thời gian thực qua Webcam
```

---

## Nhóm tuổi Phân loại

Hệ thống phân chia thành **3 nhóm tuổi** chính:
1. **Dưới 18 tuổi** (Nhãn 0)
2. **18 đến 55 tuổi** (Nhãn 1)
3. **Trên 55 tuổi** (Nhãn 2)

---

## Hướng dẫn cài đặt

1. **Clone repository này về máy:**
   ```bash
   git clone https://github.com/NQPDong/Age-Estimation.git
   cd Age-Estimation
   ```

2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chuẩn bị dữ liệu:**
   - Tải dataset UTKFace và giải nén các ảnh vào thư mục `dataset/UTKFace/`.

---

## Cách chạy dự án

### 1. Huấn luyện mô hình
Chạy script huấn luyện để tự động phát hiện, cắt khuôn mặt và thực hiện quá trình huấn luyện:
```bash
python src/train.py
```
*Mô hình có checkpoint tốt nhất sẽ được lưu tại `models/best_resnet50_utkface.pth`.*

### 2. Đánh giá mô hình
Đánh giá hiệu năng của mô hình đã huấn luyện trên tập kiểm thử:
```bash
python src/evaluate.py
```
*Các biểu đồ ROC, Confusion Matrix và Độ chính xác theo nhóm tuổi sẽ được lưu trong thư mục `results/`.*

### 3. Nhận diện ảnh tĩnh
Để dự đoán nhóm tuổi từ một bức ảnh cụ thể:
1. Mở file `src/predict.py`, chỉnh sửa đường dẫn ảnh cần test ở dòng cuối cùng:
   ```python
   test_img = os.path.join(BASE_DIR, "test", "ten_anh_cua_ban.jpg")
   ```
2. Chạy lệnh:
   ```bash
   python src/predict.py
   ```

### 4. Nhận diện qua Webcam (Thời gian thực)
Để bật webcam và nhận diện thực tế:
```bash
python src/cam.py
```
*(Nhấn phím **ESC** để thoát ứng dụng webcam).*