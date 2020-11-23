class Connection:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post


class STDPConnection(Connection):
    def __init__(self, pre, post):
        super().__init__(pre, post)


if __name__ == '__main__':
    print(Connection.mro())