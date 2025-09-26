from datasets import load_dataset

dataset = load_dataset("HuggingFaceM4/FairFace", "1.25")

for i in range(5):
    sample = dataset["train"][i]
    img = sample["image"]
    race = sample["race"] 
    age = sample['age']
    gender = sample['gender']
    
    # Save to local disk
    filename = f"fairface_{i}_{race}_{age}_{gender}.jpg"
    img.save(filename)
    print(f"Saved {filename}")