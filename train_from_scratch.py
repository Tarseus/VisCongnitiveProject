import torch 
from torchvision import datasets
from torchvision.transforms import transforms
import csv
import torch.nn as nn
import vision_transformer
from tqdm import tqdm

if __name__ == '__main__':
    model = vision_transformer.vit_small_patch16_224()
    model_name = 'vit_small_patch16_224_cifar100_scratch.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    with open('training_results_fromscratch.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Train Accuracy', 'Validation Accuracy'])

    min_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 30
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        with open('training_results_fromscratch.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_loss, train_accuracy, accuracy])
        
        if avg_loss < min_loss:
            min_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print(f'Early stopping at epoch {epoch+1}')
                break
        print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% Validation Accuracy: {accuracy:.2f}%, Min Loss: {min_loss:.4f}')
        torch.save(model.state_dict(), model_name)