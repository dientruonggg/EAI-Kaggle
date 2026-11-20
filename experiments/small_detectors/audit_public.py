"""Count Yolo-FastestV2's published optimized ncnn graph without loading a pickle."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

SOURCE = "https://raw.githubusercontent.com/dog-qiuqiu/Yolo-FastestV2/main/sample/ncnn/model/yolo-fastestv2-opt.param"
NORMALIZED_SHA256 = "8626066252cd595094d5e292271d754a382cea58a69de9529421b8281201ddef"
# (kernel values, bias values, number of layers), measured from the official graph.
CONVOLUTIONS = [[216,3,2],
    [216,24,5],
    [432,48,9],
    [576,24,9],
    [648,24,1],
    [864,12,2],
    [864,96,5],
    [1800,72,8],
    [2304,48,17],
    [5184,72,8],
    [5760,80,2],
    [9216,96,9],
    [13824,72,1],
    [20736,72,1]]


def count_convolutions(path):
    lines = [" ".join(line.split()) for line in path.read_text().splitlines() if line.strip()]
    normalized = ("\n".join(lines) + "\n").encode()
    digest = hashlib.sha256(normalized).hexdigest()
    if digest != NORMALIZED_SHA256:
        raise ValueError(f"Unexpected graph digest: {digest}")
    assert lines[0] == "7767517" and int(lines[1].split()[0]) == len(lines) - 2
    counts = Counter()
    for line in lines[2:]:
        layer, _, input_count, output_count, *fields = line.split()
        if layer in {"Convolution", "ConvolutionDepthWise"}:
            params = dict(field.split("=", 1) for field in fields[int(input_count) + int(output_count):])
            counts[int(params["6"]), int(params["0"]) if params.get("5") == "1" else 0] += 1
    return sorted([weights, bias, count] for (weights, bias), count in counts.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog=f"Official source: {SOURCE}")
    parser.add_argument("--param", type=Path, help="Verify the upstream graph against the recorded census")
    args = parser.parse_args()
    groups = CONVOLUTIONS if args.param is None else count_convolutions(args.param)
    assert groups == CONVOLUTIONS
    weights = sum(w * n for w, _, n in groups)
    biases = sum(b * n for _, b, n in groups)
    print(f"Yolo-FastestV2 ncnn graph: {weights} kernels + {biases} biases = {weights + biases}; cap 200000")
    print(f"Normalized source SHA-256: {NORMALIZED_SHA256}")
