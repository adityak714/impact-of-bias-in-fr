import marimo

__generated_with = "0.16.1"
app = marimo.App(width="columns")


@app.cell
def _():
    import cv2
    import matplotlib.pyplot as plt
    import polars as pl
    from PIL import Image
    import os
    return Image, pl


@app.cell
def _():
    from deepface import DeepFace
    return


@app.cell
def _(pl):
    # https://github.com/dchen236/FairFace?tab=readme-ov-file

    # Problem with below code: Rate Limited By HF
    # splits = {'train': 'margin025/train-*.parquet', 'val': 'margin025/val-00000-of-00001.parquet'}
    # df = pl.read_parquet('hf://datasets/ryanramos/fairface/' + splits['train'])

    df = pl.read_parquet("data/val-00000-of-00001.parquet")
    return (df,)


@app.cell
def _(df):
    print(df)
    return


@app.cell
def _(df):
    # Unnest path variable, and the binary form of the image into two separate columns
    df2 = df.unnest("file")
    df2[0,0]
    return (df2,)


@app.cell
def _(Image, df2):
    img_in_bytes: bytes = df2[0,0]
    path = "data/" + df2[0,1]
    with open(path, 'wb') as converter:
        converter.write(img_in_bytes)

    img = Image.open(path)
    # os.remove(bin2img)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
