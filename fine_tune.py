import torch 
from torchvision import datasets
from torchvision.transforms import transforms
import csv
import torch.nn as nn
import vision_transformer
from tqdm import tqdm

def transform_dict(pre_train_dict):
    new_dict = {}
    for key in pre_train_dict.keys():
        # print(key)
        if key.startswith('cls_token'):
            new_key = key.replace('cls_token', 'classification_token')
            new_dict[new_key] = pre_train_dict[key]
        elif key.startswith('pos_embed'):
            new_key = key.replace('pos_embed', 'position_embedding')
            new_dict[new_key] = pre_train_dict[key]
        elif key.startswith('patch_embed'):
            new_key = key.replace('patch_embed', 'patch_embedding')
            new_dict[new_key] = pre_train_dict[key]
        elif key.startswith('blocks'):
            new_key = key.replace('blocks', 'transformer_blocks')
            new_dict[new_key] = pre_train_dict[key]
        elif key.startswith('norm'):
            new_key = key.replace('norm', 'normalization')
            new_dict[new_key] = pre_train_dict[key]
        elif key.startswith('head'):
            new_key = key.replace('head', 'output_head')
            new_dict[new_key] = pre_train_dict[key]
    return new_dict

if __name__ == '__main__':
    model = vision_transformer.vit_small_patch16_224()
    # model.head = nn.Linear(model.head.in_features, 100)
    pre_train_dict = torch.load('vit_small_patch16_224_in1k.pth')
    model.load_state_dict(transform_dict(pre_train_dict))
    # model.load_state_dict(torch.load('vit_small_patch16_224.pth'))

    print('Pretrained model loaded.')

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
    with open('training_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Train Accuracy', 'Validation Accuracy'])
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct_train = 0  # 新增：用于跟踪训练集上的正确预测数
        total_train = 0  # 新增：用于跟踪训练集上的总样本数
        # 使用tqdm来包装enumerate，添加进度条
        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # 新增：获取最大概率的预测结果
            total_train += labels.size(0)  # 新增：累加批次中的样本数
            correct_train += (predicted == labels).sum().item()  # 新增：累加正确预测的数量

        avg_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train  # 新增：计算训练集上的准确率

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

        with open('training_results.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_loss, train_accuracy, accuracy])

        print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% Validation Accuracy: {accuracy:.2f}%')
    torch.save(model.state_dict(), 'vit_small_patch16_224_cifar100_pretrain.pth')