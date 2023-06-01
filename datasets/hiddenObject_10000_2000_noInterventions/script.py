#~/apps/blender-2.93.5-linux-x64/blender hiddenObject.blend --python ./script.py -- DATA_PATH SAVE_PATH
#blender hiddenObject.blend --python ./script.py -- DATA_PATH "./renders/" [None, Int]
# generated with Blender 2.93.5
import bpy
import os
import sys
import numpy as np

# arguments: DATA_PATH SAVE_PATH

np.random.seed(123)

colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 0.5, 0.0, 1.0)]
y_positions = [1, 3, 6]
shape_0 = [bpy.data.objects["Cube0"], bpy.data.objects["Cylinder0"], bpy.data.objects["Sphere0"], bpy.data.objects["Cone0"]]
shape_1 = [bpy.data.objects["Cube1"], bpy.data.objects["Cylinder1"], bpy.data.objects["Sphere1"], bpy.data.objects["Cone1"]]
#shape_2 = [bpy.data.objects["Cube2"], bpy.data.objects["Cylinder2"], bpy.data.objects["Sphere2"], bpy.data.objects["Cone2"]] # unobserved

#obj.data.materials.append(m0)

#Cube0,1,2
#Cylinder0
#Sphere
#Cone

#blender ignores all args after "--"
script_argv = sys.argv[sys.argv.index("--") + 1:]
print("args", script_argv)
data_path = script_argv[0]
output_dir = script_argv[1] #"./renders/"
max_samples = script_argv[2] #None or int

m0 = bpy.data.materials.get("Material0")
m1 = bpy.data.materials.get("Material1")
m2 = bpy.data.materials.get("Material2")

#obj = bpy.data.objects[f"Cone{0}"]
#obj.location.x = 0


#for i, o in enumerate(m0.node_tree.nodes["Principled BSDF"].inputs):
#  print(i, o.name)


data = np.genfromtxt(data_path, delimiter=',',dtype=np.int)
# data format [N, 10]
# each sample is a tuple (ap, ac, as, bp, bc, bs, cp, cc, cs)


if max_samples == "None":
  max_samples = len(data)
else:
  max_samples = int(max_samples)
print(f"Rendering {max_samples} of total {len(data)} samples.")

for n in range(max_samples):
    #if n % 1000 == 0:
    #    print(f"Rendering sample {n}")
    (oap, oac, oas, obp, obc, obs, ocp, occ, ocs) = data[n, :]
    
    o0 = shape_0[oas]
    o1 = shape_1[obs]
    
    #setup object 0
    o0.location.x = 1
    o0.location.y = y_positions[oap] + np.random.normal(0, 0.2)
    
    m0.node_tree.nodes["Principled BSDF"].inputs[4].default_value = 1.0 # metallic
    m0.node_tree.nodes["Principled BSDF"].inputs[0].default_value = colors[oac]
    o0.active_material = m0
    
    #setup object 1
    o1.location.x = -1
    o1.location.y = y_positions[obp] + np.random.normal(0, 0.2)
    
    m1.node_tree.nodes["Principled BSDF"].inputs[4].default_value = 1.0 # metallic
    m1.node_tree.nodes["Principled BSDF"].inputs[0].default_value = colors[obc]
    o1.active_material = m1
    
    
    #render
    img_id = n
    #img_size = [180, 120]
    bpy.context.scene.render.image_settings.file_format='JPEG'
    bpy.context.scene.render.filepath = os.path.join(output_dir, f"{img_id}")
    #bpy.context.scene.render.file_extension = ".jpg"
    #bpy.context.scene.render.resolution_x = img_size[0]
    #bpy.context.scene.render.resolution_y = img_size[1]
    bpy.ops.render.render(write_still = True)
    
    #move objects out of view
    o0.location.y = 50
    o1.location.y = 50


bpy.ops.wm.quit_blender()
