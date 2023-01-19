Thử nghiệm để trả lời nhũng câu hỏi dưới về mô hình ngôn ngữ lớn (GPT-3, PaLM ...). Bước đầu tiên là thiết lập các thử nghiệm theo [nanoGPT](https://github.com/karpathy/nanoGPT) và [cramming](https://github.com/JonasGeiping/cramming) (sao chép 1:1). Sau khi sao chép thành công sẽ áp dụng lên bộ dữ liệu càng thuần Việt càng tốt, có thể hoàn toàn là âm tiết tiếng Việt, mục đích là để tiết kiệm bộ tham số và làm nổi bật đặc trưng của tiếng Việt.

- Liệu có thể lặp lại scaling law chỉ với một lượng dữ liệu và tính toán hạn chế? (xem cramming paper)

- Liệu có thể lặp lại scaling law chỉ với một tác vụ nhỏ trong xử lý ngôn ngữ? (xem santacoder)

- Làm sao để tăng khả năng sử dụng tối đa sức mạnh phần cứng đang có để huấn luyện mô hình?
  - FlashAttention
  - AMP: Auto-Mixed Precision
  - Sử dụng [2:4 spare matrix](https://timdettmers.com/2023/01/16/which-gpu-for-deep-learning/#Sparse_Network_Training) (có thể coi đây là Dropout với p = 0.5)
  - Viết lại bằng C++/CUDA framework (tham khảo [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn))

- Các cách khác nhau để khai thác mô hình mà chưa cần fine-tune?

- Các cách khác nhau để tăng độ hiệu quả của một mô hình? (tiếp tục pre-train, fine-tune cho từng tác vụ, RLHL ...)

- Bao nhiêu lượng dữ liệu là đủ để pre-train tiếp một mô hình đang có cho một ngôn ngữ lạ?

- Liệu những gì nó học được từ ngôn ngữ này có thể "mang sang" ngôn ngữ khác không?

- Với một lượng dữ liệu nhất định, của một domain cụ thể thì nên tokenization như thế nào? Bao nhiêu params / training bao lâu là đủ?

- - -

## Tại sao cần pre-train cho riêng tiếng Việt?

Các mô hình ngôn ngữ lớn hiện nay bị thống trị bởi tiếng Anh và các ngôn ngữ gốc La-tinh, ngôn ngữ Việt do dữ liệu ít và đặc trưng riêng (các ký tự utf-8 mã hóa 2-4 byte) nên khi tokenization sẽ trở nên lép vế (xem hình dưới). Từ đấy dẫn tới thiệt hại về cả hiệu suất và kinh tế (nhiều tokens / words thì sinh câu chậm hơn, tốn tài nguyên hơn)

![](docs/files/short.jpg)
![](docs/files/long.jpg)
## Others

![](docs/files/gpt-00.jpg)
