from deepface import DeepFace

objs = DeepFace.analyze(
  img_path = "fairface_0_0_6_0.jpg", actions = ['age'], enforce_detection=False
)

print(objs)