# Experimental protocol / Phiếu ghi nhận thí nghiệm

## Giữ cố định
- Model ID và model revision (nếu pin revision)
- PyTorch, Transformers, Python versions
- Device (`mps` hoặc `cpu`)
- Warm-up, repeats, seed
- Máy cắm nguồn hay dùng pin; đóng ứng dụng nặng; tránh nhiệt độ/thermal throttling khác nhau

## Đổi một biến mỗi thí nghiệm
- A: prompt tokens = 32, 128, 256; new tokens giữ 16
- B: new tokens = 16, 32, 64; prompt giữ 64
- C: device CPU vs MPS; cùng model, prompt, new tokens và repeats

## Báo cáo
Với mỗi cấu hình, báo cáo median và standard deviation của:
- Prefill latency (s)
- Decode latency (s)
- Decode step latency (s)
- Decode tokens/s
- Process RSS (MB)

Ghi rõ lỗi, lần chạy bị gián đoạn, và không loại bỏ outlier nếu không nêu trước tiêu chí.
