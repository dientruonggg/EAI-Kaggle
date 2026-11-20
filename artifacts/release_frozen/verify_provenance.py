"""Compare frozen assets with the extracted BTC package, its ZIP and its notebook."""

import ast
import base64
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

RELEASE = Path(__file__).resolve().parent
BTC = Path("submission_package_btc")
NOTEBOOK = BTC / "DT2_EdgeAI2026_Qualifiers_Submission_98506_SOTA.ipynb"
ASSETS = {
    "DET_TFLITE_PATH": "dtnet_nwd_tal_full6k_int8.tflite",
    "INTENT_TFLITE_PATH": "text_cnn_int8.tflite",
    "INTENT_VOCAB_PATH": "text_cnn_vocab.json",
}


def notebook_assets(notebook_bytes):
    cell = json.loads(notebook_bytes)["cells"][6]
    assets = {}
    for statement in ast.parse("".join(cell["source"])).body:
        match statement:
            case ast.If(
                test=ast.UnaryOp(operand=ast.Call(func=ast.Attribute(value=ast.Name(id=variable), attr="exists"))),
                body=[ast.Assign(targets=[ast.Name(id="raw_b64")], value=payload), *_],
            ):
                assets[ASSETS[variable]] = zlib.decompress(base64.b64decode(ast.literal_eval(payload)))
            case ast.Assign(targets=[ast.Name(id="VOCAB_DICT")], value=vocabulary):
                assets[ASSETS["INTENT_VOCAB_PATH"]] = json.dumps(
                    ast.literal_eval(vocabulary), ensure_ascii=False, indent=2
                ).encode("utf-8")
    return assets


def main():
    workspace = RELEASE.parent
    notebook_bytes = (workspace / NOTEBOOK).read_bytes()
    with zipfile.ZipFile(workspace / BTC.with_suffix(".zip")) as archive:
        if archive.read(NOTEBOOK.as_posix()) != notebook_bytes:
            raise SystemExit("BTC notebook differs from ZIP")
        embedded = notebook_assets(notebook_bytes)
        for name in ASSETS.values():
            relative = Path("weights") / name
            frozen = (RELEASE / relative).read_bytes()
            package_asset = BTC / relative
            extracted = (workspace / package_asset).read_bytes()
            archived = archive.read(package_asset.as_posix())
            if frozen != extracted or frozen != archived or frozen != embedded[name]:
                raise SystemExit(f"Asset differs across sources: {relative}")
            print(f"{hashlib.sha256(frozen).hexdigest()}  {relative}  ({len(frozen):,} bytes)")


if __name__ == "__main__":
    main()
