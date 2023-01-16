Đây là thử nghiệm trả lời câu hỏi dưới đây sau khi tìm hiểu về mô hình ngôn ngữ lớn (GPT-3, PaLM ...). Bước đầu tiên là thiết lập các thử nghiệm theo https://github.com/karpathy/nanoGPT và https://github.com/JonasGeiping/cramming (sao chép 1:1). Sau khi sao chép thành công sẽ áp dụng lên bộ dữ liệu thuần Việt có thể hoàn toàn là âm tiết tiếng Việt (a domain specific dataset).

- Liệu có thể lặp lại scaling law chỉ với một lượng dữ liệu và tính toán hạn chế? (xem cramming paper)

- Liệu có thể lặp lại scaling law chỉ với một tác vụ nhỏ trong xử lý ngôn ngữ? (xem santacoder)

- Làm sao để tăng khả năng sử dụng tối đa sức mạnh phần cứng đang có để huấn luyện mô hình?
  - FlashAttention
  - AMP: Auto-Mixed Precision
  - Sử dụng [2:4 spare matrix](https://timdettmers.com/2023/01/16/which-gpu-for-deep-learning/#Sparse_Network_Training) (có thể coi đây là Dropout với p = 0.5)
  - Viết lại bằng C++/CUDA framework https://github.com/NVlabs/tiny-cuda-nn

- Các cách khác nhau để khai thác mô hình mà chưa cần fine-tune?

- Các cách khác nhau để tăng độ hiệu quả của một mô hình? (tiếp tục pre-train, fine-tune cho từng tác vụ, RLHL ...)

- Bao nhiêu lượng dữ liệu là đủ để pre-train tiếp một mô hình đang có cho một ngôn ngữ lạ?

- Liệu những gì nó học được từ ngôn ngữ này có thể "mang sang" ngôn ngữ khác không?
