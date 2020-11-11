class Connection:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post


if __name__ == '__main__':
    print(Connection.mro())