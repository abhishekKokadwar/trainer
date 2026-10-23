def train_mnist():
    import os
    import torch
    import torch.nn.functional as F
    from torch import nn, optim
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from torchvision import datasets, transforms

    # Define the PyTorch CNN model to be trained
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    # Use NCCL if a GPU is available, otherwise use Gloo as communication backend.
    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    print(f"Using Device: {device}, Backend: {backend}")

    # Setup PyTorch distributed.
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    dist.init_process_group(backend=backend)
    print(
        "Distributed Training for WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}".format(
            dist.get_world_size(),
            dist.get_rank(),
            local_rank,
        )
    )

    # Create the model and load it into the device.
    device = torch.device(f"{device}:{local_rank}")
    model=Net().to(device)
    if torch.cuda.is_available():
        model=torch.compile(model)
    model = nn.parallel.DistributedDataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Download FashionMNIST dataset only on global rank 0 process.
    if dist.get_rank() == 0:
        datasets.FashionMNIST(
            "./data",
            train=True,
            download=True,
        )
    dist.barrier()
    dataset = datasets.FashionMNIST(
        "./data",
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    # Shard the dataset across workers.
    train_loader = DataLoader(
        dataset,
        batch_size=100,
        sampler=DistributedSampler(dataset)
    )

    dist.barrier()
    for epoch in range(1, 3):
        model.train()
        
        # Iterate over mini-batches from the training set
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = F.nll_loss(outputs, target)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    # Wait for the distributed training to complete
    dist.barrier()
    if dist.get_rank() == 0:
        print("Training is finished")
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # Finally clean up PyTorch distributed
    dist.destroy_process_group()

train_mnist()
