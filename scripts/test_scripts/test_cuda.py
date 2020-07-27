import torch
import torchvision

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    x = [torch.rand(3, 300, 400).to(device), torch.rand(3, 500, 400).to(device)]
    predictions = model(x)

    print(predictions)