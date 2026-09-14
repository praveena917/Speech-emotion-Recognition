import os

dataset_path = "dataset/RAVDESS"

actors = os.listdir(dataset_path)

print("Number of actor folders:", len(actors))
print("Actor folders:", actors[:5])

first_actor = os.path.join(dataset_path, actors[0])
files = os.listdir(first_actor)

print("\nNumber of audio files:", len(files))
print("First 5 audio files:")

for file in files[:5]:
    print(file)