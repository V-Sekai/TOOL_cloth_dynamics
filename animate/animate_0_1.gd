extends SceneTree

func parse_obj(file_path: String) -> Array:
    var vertices = []
    var indices = []

    var file = FileAccess.open(file_path, FileAccess.READ)
    if file:
        while not file.eof_reached():
            var line = file.get_line()
            if line.begins_with("v "):
                var vertex = line.strip_edges().split(" ")[1:]
                vertices.append(vertex.map(float))
            elif line.begins_with("f "):
                var face_indices = []
                for part in line.strip_edges().split(" ")[1:]:
                    face_indices.append(int(part.split("/")[0]) - 1)
                if face_indices.size() == 3:
                    indices.append_array(face_indices)
                elif face_indices.size() == 4:
                    indices.append_array([face_indices[0], face_indices[1], face_indices[2]])
                    indices.append_array([face_indices[0], face_indices[2], face_indices[3]])
        file.close()

    return [vertices, indices]

func create_gltf_with_animated_blend_shapes(static_obj_file: String, base_obj_file: String, blend_shape_files: Array, animation_hz: float, output_gltf: String) -> void:
    var static_vertices, static_indices = parse_obj(static_obj_file)
    var base_vertices, base_indices = parse_obj(base_obj_file)

    if static_indices != base_indices:
        push_error("Static and base OBJ files must have the same indices")
        return

    var blend_shapes = []
    for blend_file in blend_shape_files:
        var blend_vertices, blend_indices = parse_obj(blend_file)
        if blend_indices != base_indices:
            push_error("Index mismatch in " + blend_file)
            return
        if blend_vertices.size() != base_vertices.size():
            push_error("Vertex count mismatch in " + blend_file)
            return
        var delta_vertices = []
        for i in range(blend_vertices.size()):
            var delta = []
            for j in range(blend_vertices[i].size()):
                delta.append(blend_vertices[i][j] - base_vertices[i][j])
            delta_vertices.append(delta)
        blend_shapes.append(delta_vertices)

    var mesh = Mesh.new()
    var surface_tool = SurfaceTool.new()
    surface_tool.begin(Mesh.PRIMITIVE_TRIANGLES)

    for vertex in static_vertices:
        surface_tool.add_vertex(Vector3(vertex[0], vertex[1], vertex[2]))

    for index in static_indices:
        surface_tool.add_index(index)

    mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, surface_tool.commit_to_arrays())

    for blend_shape in blend_shapes:
        var blend_shape_name = "blend_shape_" + str(blend_shapes.find(blend_shape))
        mesh.add_blend_shape(blend_shape_name)
        for i in range(blend_shape.size()):
            mesh.set_blend_shape_mode(Mesh.BLEND_SHAPE_MODE_NORMALIZED)
            mesh.set_blend_shape_vertices(blend_shape_name, i, Vector3(blend_shape[i][0], blend_shape[i][1], blend_shape[i][2]))

    var gltf_document = GLTFDocument.new()
    var root_node = Node3D.new()
    var mesh_instance = MeshInstance3D.new()
    mesh_instance.mesh = mesh
    root_node.add_child(mesh_instance)

    var animation = Animation.new()
    var track = animation.add_track(Animation.TYPE_VALUE)
    animation.track_set_path(track, "parameters/blend_shapes")
    var frame_interval = 1.0 / animation_hz

    for i in range(blend_shapes.size()):
        animation.track_insert_key(track, i * frame_interval, [1.0 if j == i else 0.0 for j in range(blend_shapes.size())])

    var animation_player = AnimationPlayer.new()
    animation_player.add_animation("blend_shape_animation", animation)
    root_node.add_child(animation_player)

    var state = GLTFState.new()
    gltf_document.append_from_scene(root_node, state)
    gltf_document.write_to_filesystem(state, output_gltf)

    print("GLTF file saved to: " + output_gltf)


func _init() -> void:
    var static_obj = "iter23/1-SPHERE.obj"
    var base_obj = "iter23/0.obj"
    var blend_shape_files = []
    for i in range(3):
        blend_shape_files.append("iter23/" + str(i) + ".obj")
    var output_file = "animated_blend_shapes.gltf"

    var animation_frequency = 10

    create_gltf_with_animated_blend_shapes(static_obj, base_obj, blend_shape_files, animation_frequency, output_file)
    quit()
