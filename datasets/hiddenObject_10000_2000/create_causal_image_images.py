import os

dataset_base_name = "causalImage"

dataset_path = "./"
print(f"Rendering dataset from {dataset_path}")

# --background  removes windows but makes output verbose
os.system(f'~/apps/blender-2.93.5-linux-x64/blender hiddenObject.blend --python ./script.py -- "{dataset_path}/train_data.csv" "./train_renders/" None')
os.system(f'~/apps/blender-2.93.5-linux-x64/blender hiddenObject.blend --python ./script.py -- "{dataset_path}/test_data.csv" "./test_renders/" None')


print("done")
