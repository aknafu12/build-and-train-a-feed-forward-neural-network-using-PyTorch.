import torch
from data.download_mnist_datasets import download_mnist_datasets
from data.data_loader import create_data_loader
from models.feedforward import FeedForwardNet
from training.train import train

EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128

if __name__ == "__main__":
    # download data and create data loader
    train_data, _ = download_mnist_datasets()
    train_dataloader = create_data_loader(train_data,BATCH_SIZE)

    # construct model and assign it to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    feed_forward_net = FeedForwardNet().to(device)
    print(feed_forward_net)

    # initialise loss funtion + optimiser
    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")
