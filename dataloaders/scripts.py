import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--gpus", type=str, default="0")
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()
print(args.gpus)
print(args.batch_size)