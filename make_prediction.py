import torch
from data.download_mnist_datasets import  download_mnist_datasets
from data.data_loader import create_data_loader
from models.feedforward import FeedForwardNet
from training.inference import predict

MODEL_PATH = "feedforwardnet.pth"
# BATCH_SIZE = 128


if __name__ == "__main__":
    _, test_data = download_mnist_datasets()
    test_dataloader = create_data_loader(test_data, batch_size=BATCH_SIZE)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeedForwardNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Get a single test sample
    test_sample, label = next(iter(test_dataloader))

    # Predict
    prediction = predict(model, test_sample, device)
    print(f"Predicted: {prediction}, Actual: {label.item()}")
