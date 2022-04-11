
def test(a, b):
    return a + b
if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--use_efficient", default=True, action="store_true", help="use EfficientGlobalPointer")
    args = parse.parse_args()
    print(args)