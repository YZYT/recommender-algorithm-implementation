class C1:
    def __init__(self, x):
        self.x = x

    def pp(self):
        print("GGG")


class C2(C1):
    def __init__(self, x):
        super().__init__(x)


if __name__ == '__main__':
    with open("data.txt") as file:
        lines = file.readlines()
    print(lines)
