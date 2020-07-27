import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('hhhhh')
    else:
        print('oh no!')