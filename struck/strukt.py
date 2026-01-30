
class xywh:
    def __init__(self,x=0,y=0,w=0,h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class CameraStream:
    def __init__(self, depth_frame=0, depth_intrinsics=0,color_intrinsics=0,depth_image=0,color_image=0):
        self.depth_frame = depth_frame
        self.depth_intrin = depth_intrinsics
        self.color_intrin = color_intrinsics
        self.depth_image = depth_image
        self.color_image = color_image

class point:
    def __init__(self,x=0,y=0,z=0):
        self.x = x
        self.y = y
        self.z = z