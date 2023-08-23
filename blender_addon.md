# Blender Addon
## Selecting vertex indexes

```python
import bpy

# Ensure that we are in object mode
bpy.ops.object.mode_set(mode='OBJECT')

# Get the active object (assumes it's a mesh)
obj = bpy.context.active_object

# Get the selected vertices using list comprehension
selected_vertices = [v.index for v in obj.data.vertices if v.select]

# Print the list of selected vertices
print(selected_vertices)
```
