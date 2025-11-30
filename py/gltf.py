# MIT License
# Copyright (c) 2025 Anton Schreiner

import os, sys
from pathlib import Path

import gltflib
import numpy as np
import ctypes

from py.utils import *

class GltfUint32(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint32)]
class GltfUint16(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint16)]
class GltfFloat2(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]
class GltfFloat3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]
class GltfFloat4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float), ("w", ctypes.c_float)]

class BufferReader:
    def __init__(self, bytes : np.ndarray):
        self.bytes          = bytes
        self.offset         = 0
    
    def Read(self, offset, count, type):
        new_arr = np.zeros(count, dtype=type)
        assert self.offset + count * ctypes.sizeof(type) <= self.bytes.nbytes, "Buffer read out of bounds"
        ctypes.memmove(new_arr.ctypes.data, self.bytes.ctypes.data + self.offset + offset, count * ctypes.sizeof(type))
        return new_arr
    
    def View(self, offset, count, type):
        assert self.offset + offset + count * ctypes.sizeof(type) <= self.bytes.nbytes, "Buffer view out of bounds"
        return np.frombuffer(buffer=self.bytes, offset=offset + self.offset, count=count, dtype=type)

def map_gltf_type_to_struct(gltf_type, component_type):
    if gltf_type == gltflib.AccessorType.SCALAR and component_type == gltflib.ComponentType.UNSIGNED_INT: return GltfUint32
    if gltf_type == gltflib.AccessorType.SCALAR and component_type == gltflib.ComponentType.UNSIGNED_SHORT: return GltfUint16
    if gltf_type == gltflib.AccessorType.VEC2 and component_type == gltflib.ComponentType.FLOAT: return GltfFloat2
    if gltf_type == gltflib.AccessorType.VEC3 and component_type == gltflib.ComponentType.FLOAT: return GltfFloat3
    if gltf_type == gltflib.AccessorType.VEC4 and component_type == gltflib.ComponentType.FLOAT: return GltfFloat4
    assert False, "Unsupported type"

class AttributeReader:
    def __init__(self, gltf_scene: 'GLTFScene', accessor_index: int):
        self.gltf_scene       = gltf_scene
        self.accessor_index   = accessor_index
        self.accessor         = gltf_scene.gltf.model.accessors[accessor_index]
        self.buffer_view      = gltf_scene.gltf.model.bufferViews[self.accessor.bufferView]
        self.buffer           = gltf_scene.gltf.model.buffers[self.buffer_view.buffer]
        self.buffer_offset    = self.accessor.byteOffset if self.accessor.byteOffset is not None else 0
        self.buffer_offset    += self.buffer_view.byteOffset if self.buffer_view.byteOffset is not None else 0
        self.buffer_length    = self.buffer_view.byteLength if self.buffer_view.byteLength is not None else self.buffer.length
        self.buffer_stride    = self.buffer_view.byteStride
        self.normalized       = self.accessor.normalized
        self.component_type   = gltflib.ComponentType(self.accessor.componentType)
        self.count            = self.accessor.count
        self.type             = gltflib.AccessorType(self.accessor.type)
        self.struct_type = map_gltf_type_to_struct(self.type, self.component_type)
        assert self.buffer_stride == None or self.buffer_stride == ctypes.sizeof(self.struct_type), f"Buffer stride does not match struct size {self.buffer_stride} != {ctypes.sizeof(self.struct_type)} ({self.struct_type})"
        self.buffer_reader  = BufferReader(self.gltf_scene.LoadBuffer(self.buffer.uri))
        self.view           = self.buffer_reader.View(offset=self.buffer_offset,
                                                        # count=self.buffer_length // self.buffer_stride, type=self.struct_type)
                                                        count=self.count, type=self.struct_type)

    def Print(self):
        print("Accessor Index: ", self.accessor_index)
        print("Accessor: ", self.accessor)
        print("Buffer View: ", self.buffer_view)
        print("Buffer: ", self.buffer)
        print("Normalized: ", self.normalized)
        print("Component Type: ", self.component_type)
        print("Count: ", self.count)
        print("Type: ", self.type)

    def __repr__(self):
        return f"Accessor Index: {self.accessor_index}, Accessor: {self.accessor}, Buffer View: {self.buffer_view}, Buffer: {self.buffer}, Normalized: {self.normalized}, Component Type: {self.component_type}, Count: {self.count}, Type: {self.type}"

class GLTFPrimitive:
    def __init__(self, gltf_scene: 'GLTFScene', primitive: gltflib.Primitive):
        self.gltf_scene = gltf_scene
        self.primitive = primitive

        self.attributes = {}
        for attribute_name, accessor_index in dict(primitive.attributes.__dict__).items():
            if accessor_index is not None:
                self.attributes[attribute_name] = AttributeReader(gltf_scene, accessor_index).view

        self.indices = None
        if primitive.indices is not None:
            self.indices = AttributeReader(gltf_scene, primitive.indices).view
        else:
            assert False, "Indices are required"

        self.material = primitive.material
        if self.material is not None:
            self.material = gltf_scene.gltf.model.materials[primitive.material]

class GLTFMesh:
    def __init__(self, gltf_scene: 'GLTFScene', mesh_index: int):
        self.gltf_scene = gltf_scene
        self.mesh_index = mesh_index
        self.mesh       = gltf_scene.gltf.model.meshes[mesh_index]

        self.primitives = []
        for primitive in self.mesh.primitives:
            self.primitives.append(GLTFPrimitive(gltf_scene, primitive))

class GLTFScene:
    def LoadBuffer(self, uri):
        if uri in self.buffer_cache:
            return self.buffer_cache[uri]
        else:
            buffer_path             = self.gltf_folder / uri
            np_array                = np.fromfile(buffer_path, dtype=np.uint8)
            self.buffer_cache[uri]  = np_array
            return np_array

    def __init__(self, gltf_file: Path):
        if isinstance(gltf_file, str): gltf_file = Path(gltf_file)
        self.gltf       = gltflib.GLTF.load(gltf_file)
        self.gltf_file  = gltf_file
        self.gltf_folder = gltf_file.parent
        self.buffer_cache = {}
        
        self.meshes = []
        for mesh_index in range(len(self.gltf.model.meshes)):
            self.meshes.append(GLTFMesh(self, mesh_index))
        
        self.scenes = self.gltf.model.scenes[self.gltf.model.scene]
        self.nodes  = self.gltf.model.nodes

        return
        accessors       = self.gltf.model.accessors
        buffers         = self.gltf.model.buffers
        buffer_views    = self.gltf.model.bufferViews
        images          = self.gltf.model.images
        materials       = self.gltf.model.materials
        meshes          = self.gltf.model.meshes
        nodes           = self.gltf.model.nodes
        samplers        = self.gltf.model.samplers
        skins           = self.gltf.model.skins
        textures        = self.gltf.model.textures

        for mesh in meshes:
            for primitive in mesh.primitives:
                # print(primitive)
                POSITION        = primitive.attributes.POSITION
                NORMAL          = primitive.attributes.NORMAL
                TANGENT         = primitive.attributes.TANGENT
                TEXCOORD_0      = primitive.attributes.TEXCOORD_0
                TEXCOORD_1      = primitive.attributes.TEXCOORD_1
                COLOR_0         = primitive.attributes.COLOR_0
                JOINTS_0        = primitive.attributes.JOINTS_0
                WEIGHTS_0       = primitive.attributes.WEIGHTS_0

                indices        = primitive.indices
                material       = primitive.material

                assert POSITION is not None, "POSITION is required"
                assert NORMAL is not None, "NORMAL is required"
                # assert TANGENT is not None, "TANGENT is required"
                assert TEXCOORD_0 is not None, "TEXCOORD_0 is required"
                # assert TEXCOORD_1 is not None, "TEXCOORD_1 is required"
                # assert COLOR_0 is not None, "COLOR_0 is required"
                # assert JOINTS_0 is not None, "JOINTS_0 is required"
                # assert WEIGHTS_0 is not None, "WEIGHTS_0 is required"
                assert indices is not None, "indices is required"
                assert material is not None, "material is required"
                
                positions = AttributeReader(self, POSITION)

                assert positions.type == gltflib.AccessorType.VEC3, "POSITION must be VEC3"
                assert positions.component_type == gltflib.ComponentType.FLOAT, "POSITION must be FLOAT"

                positions_array = positions.Read()
                print(positions_array)

                normals = AttributeReader(self, NORMAL)
                assert normals.type == gltflib.AccessorType.VEC3, "NORMAL must be VEC3"
                assert normals.component_type == gltflib.ComponentType.FLOAT, "NORMAL must be FLOAT"

                # tangents = AttributeReader(scene, TANGENT)
                texcoords_0 = AttributeReader(self, TEXCOORD_0)
                assert texcoords_0.type == gltflib.AccessorType.VEC2, "TEXCOORD_0 must be VEC2"
                assert texcoords_0.component_type == gltflib.ComponentType.FLOAT, "TEXCOORD_0 must be FLOAT"
                # texcoords_1 = AttributeReader(scene, TEXCOORD_1)
                # colors_0 = AttributeReader(scene, COLOR_0)
                # joints_0 = AttributeReader(scene, JOINTS_0)
                # weights_0 = AttributeReader(scene, WEIGHTS_0)

                # print(positions)



# SCALAR = 'SCALAR'
# VEC2 = 'VEC2'
# VEC3 = 'VEC3'
# VEC4 = 'VEC4'
# MAT2 = 'MAT2'
# MAT3 = 'MAT3'
# MAT4 = 'MAT4'

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="GLTF Scene Viewer")
    parser.add_argument("--gltf_file", type=str, help="GLTF file to load")
    args = parser.parse_args()


    scene = GLTFScene(Path(args.gltf_file))
    # print(scene.gltf.model)


        # print(CONSOLE_COLOR_RED + "--------------------------------" + CONSOLE_COLOR_RESET)

    print("Done.")