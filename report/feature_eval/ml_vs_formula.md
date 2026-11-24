| Phương pháp | Acc | macro-F1 | F1 flap | Báo nhầm idle | Trễ flap (từ tay đập, ms, trung vị) | Suy luận (ms/mẫu) | Giải thích |
|---|---|---|---|---|---|---|---|
| Công thức tay (ngưỡng chọn) | 0.657 | 0.588 | 0.000 | 0.028 | nan | 0.017 | cao: mỗi lệnh = 1 công thức/ngưỡng có nghĩa vật lý |
| Logistic Regression | 0.779 | 0.763 | 0.746 | 0.085 | -5526 | 0.389 | khá: hệ số tuyến tính |
| SVM (RBF) | 0.805 | 0.794 | 0.784 | 0.098 | -5526 | 0.529 | thấp |
| Random Forest | 0.804 | 0.792 | 0.780 | 0.104 | -5526 | 29.383 | trung bình: độ quan trọng đặc trưng |
| MLP (32-16) | 0.791 | 0.779 | 0.757 | 0.107 | -5526 | 0.680 | thấp |
