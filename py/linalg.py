# MIT License
# Copyright (c) 2025 Anton Schreiner

import numpy as np
import ctypes

class f32x2(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

    def __add__(self, other): return f32x2(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return f32x2(self.x - other.x, self.y - other.y)
    def __mul__(self, other): return f32x2(self.x * other.x, self.y * other.y)
    def __truediv__(self, other): return f32x2(self.x / other.x, self.y / other.y)

class f32x3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]

    def __add__(self, other): return f32x3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other): return f32x3(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, other): return f32x3(self.x * other.x, self.y * other.y, self.z * other.z)
    def __truediv__(self, other): return f32x3(self.x / other.x, self.y / other.y, self.z / other.z)

    def cross(self, other):
        return f32x3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def normalize(self):
        l = np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        return f32x3(self.x / l, self.y / l, self.z / l)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def neg(self):
        return f32x3(-self.x, -self.y, -self.z)

class f32x4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float), ("w", ctypes.c_float)]

    def __add__(self, other): return f32x4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)
    def __sub__(self, other): return f32x4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)
    def __mul__(self, other):
        if isinstance(other, f32x4):
            return f32x4(self.x * other.x, self.y * other.y, self.z * other.z, self.w * other.w)
        elif isinstance(other, f32x4x4):
            return f32x4(
                self.x * other.r0.x + self.y * other.r1.x + self.z * other.r2.x + self.w * other.r3.x,
                self.x * other.r0.y + self.y * other.r1.y + self.z * other.r2.y + self.w * other.r3.y,
                self.x * other.r0.z + self.y * other.r1.z + self.z * other.r2.z + self.w * other.r3.z,
                self.x * other.r0.w + self.y * other.r1.w + self.z * other.r2.w + self.w * other.r3.w
            )
    def __truediv__(self, other): return f32x4(self.x / other.x, self.y / other.y, self.z / other.z, self.w / other.w)

def quaternion_to_matrix(quat):
    # Expect quat as [w, x, y, z]
    w, x, y, z = quat
    
    # Compute common terms to avoid duplication
    x2, y2, z2 = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    # Build 3x3 rotation matrix
    rotation = np.array([
        [1 - 2*(y2 + z2),     2*(xy - wz),     2*(xz + wy), 0],
        [    2*(xy + wz), 1 - 2*(x2 + z2),     2*(yz - wx), 0],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(x2 + y2), 0],
        [              0,               0,               0,  1]
    ])

    return rotation

class f32x4x4(ctypes.Structure):
    _fields_ = [("r0", f32x4), ("r1", f32x4), ("r2", f32x4), ("r3", f32x4)]

    def transpose(self):
        return f32x4x4(
            f32x4(self.r0.x, self.r1.x, self.r2.x, self.r3.x),
            f32x4(self.r0.y, self.r1.y, self.r2.y, self.r3.y),
            f32x4(self.r0.z, self.r1.z, self.r2.z, self.r3.z),
            f32x4(self.r0.w, self.r1.w, self.r2.w, self.r3.w)
        )

    def to_np(self):
        return np.array([
            [self.r0.x, self.r0.y, self.r0.z, self.r0.w],
            [self.r1.x, self.r1.y, self.r1.z, self.r1.w],
            [self.r2.x, self.r2.y, self.r2.z, self.r2.w],
            [self.r3.x, self.r3.y, self.r3.z, self.r3.w]
        ])

    def inverse(self):
        i = np.linalg.inv(self.to_np())
        return f32x4x4(
            f32x4(i[0][0], i[0][1], i[0][2], i[0][3]),
            f32x4(i[1][0], i[1][1], i[1][2], i[1][3]),
            f32x4(i[2][0], i[2][1], i[2][2], i[2][3]),
            f32x4(i[3][0], i[3][1], i[3][2], i[3][3])
        )

    def __mul__(self, other):
        if isinstance(other, f32x4):
            return f32x4(
                self.r0.x * other.x + self.r0.y * other.y + self.r0.z * other.z + self.r0.w * other.w,
                self.r1.x * other.x + self.r1.y * other.y + self.r1.z * other.z + self.r1.w * other.w,
                self.r2.x * other.x + self.r2.y * other.y + self.r2.z * other.z + self.r2.w * other.w,
                self.r3.x * other.x + self.r3.y * other.y + self.r3.z * other.z + self.r3.w * other.w
            )
        elif isinstance(other, f32x4x4):
            return f32x4x4(
                f32x4(
                    self.r0.x * other.r0.x + self.r0.y * other.r1.x + self.r0.z * other.r2.x + self.r0.w * other.r3.x,
                    self.r0.x * other.r0.y + self.r0.y * other.r1.y + self.r0.z * other.r2.y + self.r0.w * other.r3.y,
                    self.r0.x * other.r0.z + self.r0.y * other.r1.z + self.r0.z * other.r2.z + self.r0.w * other.r3.z,
                    self.r0.x * other.r0.w + self.r0.y * other.r1.w + self.r0.z * other.r2.w + self.r0.w * other.r3.w
                ),
                f32x4(
                    self.r1.x * other.r0.x + self.r1.y * other.r1.x + self.r1.z * other.r2.x + self.r1.w * other.r3.x,
                    self.r1.x * other.r0.y + self.r1.y * other.r1.y + self.r1.z * other.r2.y + self.r1.w * other.r3.y,
                    self.r1.x * other.r0.z + self.r1.y * other.r1.z + self.r1.z * other.r2.z + self.r1.w * other.r3.z,
                    self.r1.x * other.r0.w + self.r1.y * other.r1.w + self.r1.z * other.r2.w + self.r1.w * other.r3.w
                ),
                f32x4(
                    self.r2.x * other.r0.x + self.r2.y * other.r1.x + self.r2.z * other.r2.x + self.r2.w * other.r3.x,
                    self.r2.x * other.r0.y + self.r2.y * other.r1.y + self.r2.z * other.r2.y + self.r2.w * other.r3.y,
                    self.r2.x * other.r0.z + self.r2.y * other.r1.z + self.r2.z * other.r2.z + self.r2.w * other.r3.z,
                    self.r2.x * other.r0.w + self.r2.y * other.r1.w + self.r2.z * other.r2.w + self.r2.w * other.r3.w
                ),
                f32x4(
                    self.r3.x * other.r0.x + self.r3.y * other.r1.x + self.r3.z * other.r2.x + self.r3.w * other.r3.x,
                    self.r3.x * other.r0.y + self.r3.y * other.r1.y + self.r3.z * other.r2.y + self.r3.w * other.r3.y,
                    self.r3.x * other.r0.z + self.r3.y * other.r1.z + self.r3.z * other.r2.z + self.r3.w * other.r3.z,
                    self.r3.x * other.r0.w + self.r3.y * other.r1.w + self.r3.z * other.r2.w + self.r3.w * other.r3.w
                )
            )
        else:
            raise ValueError("Invalid type for multiplication")
        
    @staticmethod
    def from_quanternion(q):
        m = quaternion_to_matrix(q)
        return f32x4x4(
            f32x4(m[0][0], m[0][1], m[0][2], m[0][3]),
            f32x4(m[1][0], m[1][1], m[1][2], m[1][3]),
            f32x4(m[2][0], m[2][1], m[2][2], m[2][3]),
            f32x4(m[3][0], m[3][1], m[3][2], m[3][3])
        )
    
    @staticmethod
    def from_scale(s):
        return f32x4x4(
            f32x4(s.x, 0, 0, 0),
            f32x4(0, s.y, 0, 0),
            f32x4(0, 0, s.z, 0),
            f32x4(0, 0, 0, 1)
        )

    @staticmethod
    def from_offset(o):
        return f32x4x4(
            f32x4(1, 0, 0, 0),
            f32x4(0, 1, 0, 0),
            f32x4(0, 0, 1, 0),
            f32x4(o.x, o.y, o.z, 1)
        )


    def print(self):
        print("--------------------------------------------------------")
        print(f"r0 = {self.r0.x}, {self.r0.y}, {self.r0.z}, {self.r0.w}")
        print(f"r1 = {self.r1.x}, {self.r1.y}, {self.r1.z}, {self.r1.w}")
        print(f"r2 = {self.r2.x}, {self.r2.y}, {self.r2.z}, {self.r2.w}")
        print(f"r3 = {self.r3.x}, {self.r3.y}, {self.r3.z}, {self.r3.w}")
        print("--------------------------------------------------------")

def f32x3_splat(v): return f32x3(v, v, v)

class Camera:
    def __init__(self):
        self.pos    = f32x3(0, 0, 0)
        self.phi    = 0
        self.theta  = np.pi / 2

        self.look_at = f32x3(0, 0, 0)

        self.fov     = 80
        self.up      = f32x3(0, 1, 0)
        self.right   = f32x3(1, 0, 0)
        self.forward = f32x3(0, 0, 1)
        self.aspect  = 1

        self.y_is_up = True

        self.frustum_x = f32x3(1, 0, 0)
        self.frustum_y = f32x3(0, 1, 0)
        self.frustum_z = f32x3(0, 0, 1) 
        self.half_fov_tan = np.tan(np.radians(self.fov / 2))

    def from_json(self, data):
        self.pos = f32x3(data["pos"][0], data["pos"][1], data["pos"][2])
        self.phi = data["phi"]
        self.theta = data["theta"]
        self.fov = data["fov"]
        self.aspect = data["aspect"]
        self.y_is_up = data["y_is_up"]

    def to_json(self):
        return {
            "pos": [self.pos.x, self.pos.y, self.pos.z],
            "phi": self.phi,
            "theta": self.theta,
            "fov": self.fov,
            "aspect": self.aspect,
            "y_is_up": self.y_is_up
        }

    def update(self):

        if self.y_is_up:
            self.forward = f32x3(
                np.cos(self.theta) * np.cos(self.phi),
                np.sin(self.theta),
                np.cos(self.theta) * np.sin(self.phi),
            )

            self.right      = self.forward.cross(f32x3(0, 1, 0)).normalize()
            self.up         = self.right.cross(self.forward).normalize()
            self.look_at    = self.pos + self.forward

            self.frustum_x  = self.right * f32x3_splat(self.half_fov_tan * self.aspect)
            self.frustum_y  = self.up * f32x3_splat(self.half_fov_tan)
            self.frustum_z  = self.forward
        else: # z is up
            self.forward = f32x3(
                np.cos(self.theta) * np.cos(self.phi),
                np.cos(self.theta) * np.sin(self.phi),
                np.sin(self.theta),
            )

            self.right      = self.forward.cross(f32x3(0, 0, 1)).normalize()
            self.up         = self.right.cross(self.forward).normalize().neg()
            self.look_at    = self.pos + self.forward

            self.frustum_x  = self.right * f32x3_splat(self.half_fov_tan * self.aspect)
            self.frustum_y  = self.up * f32x3_splat(self.half_fov_tan)
            self.frustum_z  = self.forward


    def move_forward(self, dt): self.pos += self.forward * f32x3_splat(dt)
    def move_right(self, dt): self.pos += self.right * f32x3_splat(dt)
    def move_up(self, dt): self.pos += self.up * f32x3_splat(dt)

if __name__ == "__main__":
    mat = f32x4x4(
        f32x4(2, 1, 3, 4),
        f32x4(1, 1, 2, 3),
        f32x4(2, 5, 1, 2),
        f32x4(8, 3, 0, 1)
    )
    mat2 = mat.inverse()
    mat3 = mat * mat2

    mat.print()
    mat2.print()
    mat3.print()


    assert mat3.r0.x == 1 and mat3.r0.y == 0 and mat3.r0.z == 0 and mat3.r0.w == 0
    assert mat3.r1.x == 0 and mat3.r1.y == 1 and mat3.r1.z == 0 and mat3.r1.w == 0
    assert mat3.r2.x == 0 and mat3.r2.y == 0 and mat3.r2.z == 1 and mat3.r2.w == 0
    assert mat3.r3.x == 0 and mat3.r3.y == 0 and mat3.r3.z == 0 and mat3.r3.w == 1

