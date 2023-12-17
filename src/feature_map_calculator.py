from model import Net
from nn import Tensor


def main():
    net = Net()

    input_data = Tensor((8, 1, 28, 28))

    print(net.summary(input_data))


if __name__ == '__main__':
    main()
