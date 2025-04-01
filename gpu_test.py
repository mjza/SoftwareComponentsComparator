import torch

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device:", torch.cuda.get_device_name(device))

        # Simple tensor operation on GPU
        x = torch.rand(1000, 1000, device=device)
        y = torch.rand(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print("Matrix multiplication result shape:", z.shape)
    else:
        print("CUDA not available. Using CPU.")

if __name__ == "__main__":
    main()
