import torch

print("torch", torch.__version__)
print(
    "MPS built:",
    torch.backends.mps.is_built(),
    "| available:",
    torch.backends.mps.is_available(),
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(2048, 20248, device=device)
print("matmul su", device, "->", (x @ x).mean().item())
