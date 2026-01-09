# Author: Mark Smithson
# File to run all the FairFace inferences on Modal due to the fact of how compute intensive it is.

import modal

# Define the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    .pip_install(
        "tensorflow==2.15.0",  # Older TF that works better with deepface
        "tf-keras",
        "deepface",
        "datasets",
        "pandas",
        "numpy",
        "pillow",
        "tqdm",
        "opencv-python-headless",
    )
)

app = modal.App("deepface-fairface-analysis", image=image)

# Create a volume to persist results
volume = modal.Volume.from_name("deepface-results", create_if_missing=True)

VOLUME_PATH = "/results"


def augment_and_save(image_pil, index, race, age, gender, output_dir):
    """Applies augmentations and saves directly."""
    import os
    import numpy as np
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter

    os.makedirs(output_dir, exist_ok=True)
    base_fn = f"fairface_{index}_{race}_{age}_{gender}_val"

    def save(img, suffix):
        # Convert grayscale back to RGB for consistency
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(os.path.join(output_dir, f"{base_fn}_{suffix}.jpg"))

    # Augmentations (Brightness, Blur, Grayscale, Noise, Rotation)
    enhancer = ImageEnhance.Brightness(image_pil)

    save(enhancer.enhance(0.7), "bright0.7")
    save(enhancer.enhance(1.3), "bright1.3")

    save(image_pil.filter(ImageFilter.GaussianBlur(radius=2)), "blur")

    save(ImageOps.grayscale(image_pil), "gray")

    img_arr = np.array(image_pil)
    noise = np.random.normal(0, 10, img_arr.shape)
    noisy_img = Image.fromarray(np.clip(img_arr + noise, 0, 255).astype("uint8"))
    save(noisy_img, "noise")

    save(image_pil.rotate(-15), "rot-15")
    save(image_pil.rotate(15), "rot15")


@app.function(timeout=3600, volumes={VOLUME_PATH: volume})
def prepare_dataset(test_mode: bool = True, num_test_images: int = 10):
    """Download and augment the FairFace validation dataset."""
    import os
    from datasets import load_dataset
    from tqdm import tqdm

    suffix = "_test" if test_mode else ""
    output_dir = f"{VOLUME_PATH}/validation_top5{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already prepared (7 augmentations per image) 
    # This is needed as we are running this multiple times and in parallel
    existing_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
    expected_files = num_test_images * 7 if test_mode else 1000  # 7 augmentations per image
    
    if len(existing_files) >= expected_files:
        return output_dir

    print("Loading FairFace dataset...")
    dataset = load_dataset("HuggingFaceM4/FairFace", "0.25")
    val_data = dataset["validation"]

    num_images = num_test_images if test_mode else len(val_data)
    print(f"{'TEST MODE: ' if test_mode else ''}Processing {num_images} images with augmentations...")

    for i in tqdm(range(num_images)):
        sample = val_data[i]
        img = sample["image"]

        # Ensure image is in RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")

        augment_and_save(
            image_pil=img,
            index=i,
            race=sample["race"],
            age=sample["age"],
            gender=sample["gender"],
            output_dir=output_dir,
        )

    volume.commit()
    print("Done preparing dataset!")
    return output_dir


# Single function for all backends. Using Modal's GPU is expensive and couldn't manage to get it to work
# This params ensure that I run all the experiment inside free credits
@app.function(
    timeout=43200,  # 12 hours
    volumes={VOLUME_PATH: volume},
    cpu=4,
    memory=8192,
)
def analyze_with_backend(backend: str, image_folder: str, max_images: int = None):
    return _analyze_backend(backend, image_folder, max_images)


def _analyze_backend(backend: str, image_folder: str, max_images: int = None):
    import os
    import pandas as pd
    from deepface import DeepFace
    from tqdm import tqdm
    import time

    print(f"Analyzing with backend: {backend}")

    results = []
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
    
    if max_images:
        image_files = image_files[:max_images]

    total_images = len(image_files)
    print(f"Total images: {total_images}")
    
    start_time = time.time()

    for idx, fname in enumerate(tqdm(image_files)):
        img_path = os.path.join(image_folder, fname)
        try:
            analysis = DeepFace.analyze(
                detector_backend=backend,
                align=True,
                img_path=img_path,
                actions=["gender", "race", "age"],
                enforce_detection=False,
            )

            result = analysis[0]

            results.append(
                {
                    "filename": fname,
                    "predicted_gender": result["dominant_gender"],
                    "gender_confidence": result["gender"],
                    "predicted_race": result["dominant_race"],
                    "race_confidence": result["race"],
                    "predicted_age": result["age"],
                }
            )
            
            # Print every 100 images
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (total_images - idx - 1) / rate
                print(f"[{backend}] {idx + 1}/{total_images} | {rate:.1f} img/s | ETA: {remaining/60:.1f} min")

        except Exception as e:
            print(f"Error processing {fname} with {backend}!!!")
            results.append(
                {
                    "filename": fname,
                    "predicted_gender": "error",
                    "gender_confidence": None,
                    "predicted_race": "error",
                    "race_confidence": None,
                    "predicted_age": None,
                    "error": str(e),
                }
            )

    elapsed_total = time.time() - start_time
    df = pd.DataFrame(results)
    
    is_test = "_test" in image_folder
    suffix = "_test" if is_test else ""
    output_path = f"{VOLUME_PATH}/{backend}_deepface_predictions{suffix}.csv"
    
    df.to_csv(output_path, index=False)
    volume.commit()

    print(f"\n Backend completed --> {backend}")
    print(f"Total time: {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")

    return {"backend": backend, "processed": len(results), "output_path": output_path, "time_minutes": elapsed_total/60}


@app.function(timeout=600, volumes={VOLUME_PATH: volume})
def combine_results(test_mode: bool = True):
    """Combine all backend results into a single summary."""
    import os
    import pandas as pd

    suffix = "_test" if test_mode else ""
    csv_files = [f for f in os.listdir(VOLUME_PATH) if f.endswith(f"_predictions{suffix}.csv")]

    if not csv_files:
        print("No result files found!")
        return []

    all_results = {}
    for csv_file in csv_files:
        backend = csv_file.replace(f"_deepface_predictions{suffix}.csv", "")
        df = pd.read_csv(f"{VOLUME_PATH}/{csv_file}")
        all_results[backend] = df
        print(f"{backend}: {len(df)} predictions")

    volume.commit()
    return csv_files


@app.local_entrypoint()
def main(full: bool = False):
    test_mode = not full
    
    all_backends = ["opencv", "yunet"]
    
    # Testing if it runs in modal
    if test_mode:
        backends = ["opencv"]  
        num_test_images = 10
        print("TEST MODE!!")
    else:
        backends = all_backends
        num_test_images = None
        print("FULL MODE!!")

    # Get dataset ready
    image_folder = prepare_dataset.remote(test_mode=test_mode, num_test_images=num_test_images or 10)

    # Run all backends in parallel on CPU
    max_images = num_test_images * 7 if test_mode else None
    
    # Launch all backends on CPU
    futures = []
    for backend in backends:
        print(f"   Launching {backend}...")
        futures.append(analyze_with_backend.spawn(backend, image_folder, max_images))
    
    # Wait for all results
    print("\nWaiting for all backends to complete...")
    results = []
    for future in futures:
        results.append(future.get())

    csv_files = combine_results.remote(test_mode=test_mode)

    if test_mode:
        print("TEST FINISHED!")
    else:
        print("FULL RUN FINISHED!")