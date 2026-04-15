Đồ án Nhận diện Độ Tuổi qua Khuôn Mặt (Age Estimation)

Dự án ứng dụng Deep Learning để dự đoán độ tuổi của người dùng thông qua ảnh tĩnh hoặc trực tiếp qua luồng Webcam.

Kiến trúc Hệ thống

Dataset: Tập dữ liệu UTKFace.

Mô hình (Model): Sử dụng Transfer Learning với kiến trúc ResNet50 (đóng băng các lớp chập, tinh chỉnh các lớp Dense cuối).

Công cụ xử lý ảnh: OpenCV.

Loss Function: Huber Loss.

Optimizer: Adam.

Cấu trúc Thư mục

📦 deeplearning
 ┣ 📂 dataset/UTKFace  # (Cần tải thủ công từ Kaggle)
 ┣ 📂 models           # Chứa file weights (.keras)
 ┣ 📂 results          # Biểu đồ training
 ┣ 📜 requirements.txt # Thư viện cần thiết
 ┣ 📜 README.md
 ┗ 📂 src
   ┣ 📜 cam.py         # Chạy nhận diện qua Webcam
   ┣ 📜 dataset_loader.py # Xử lý và load dữ liệu bằng tf.data
   ┣ 📜 evaluate.py    # Đánh giá mô hình trên tập test
   ┣ 📜 model.py       # Định nghĩa kiến trúc ResNet50
   ┣ 📜 predict.py     # Dự đoán ảnh tĩnh
   ┗ 📜 train.py       # Vòng lặp huấn luyện mô hình


Hướng dẫn cài đặt

Clone repository này về máy.

Tạo môi trường ảo và cài đặt thư viện:

pip install -r requirements.txt


Tạo thư mục dataset/UTKFace và đưa ảnh vào đó.

Chạy python src/train.py để huấn luyện lại từ đầu.

Chạy Ứng dụng

Để bật webcam và nhận diện thực tế, chạy lệnh sau:

python src/cam.py


(Nhấn phím ESC để thoát).