from rwkvstic.load import RWKV
# import torch; backend = "pytorch(cpu/gpu)"; kwargs = { "useGPU": False, "runtimedtype": torch.bfloat16, "dtype": torch.bfloat16} #"chunksize": 4 quant(gpu-8bit)
# model = RWKV("RWKV-4-Pile-7B-20230109-ctx4096.pth", backend, **kwargs)
backend = "jax(cpu/gpu/tpu)"; kwargs = {}
model = RWKV("RWKV-4-Pile-3B-20221110-ctx4096.pth", backend, **kwargs)
ELDR = "\n\nExpert Long Detailed Response: "
model.resetState();t=input("question: ");model.loadContext("\n", t+ELDR);print(model.forward(number=100)["output"])
