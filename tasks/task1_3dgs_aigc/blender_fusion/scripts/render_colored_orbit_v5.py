import bpy
import math
import mathutils
import os

print("🚀 启动【Blender 4.0 电影级运镜】150帧多段全彩环绕视频管线...")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(TASK_DIR, "assets", "input-assets")
OUTPUT_DIR = os.path.join(TASK_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 一、初始化与彻底清空
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 12  # 低采样确保多帧数下依然能快速出图

scene.render.resolution_x = 1280
scene.render.resolution_y = 720

# 🎯 核心改动 1：总帧数提升至 150 帧，让旋转速度明显慢下来，同时给推拉镜头留足空间
FRAME_COUNT = 150
scene.frame_start = 1
scene.frame_end = FRAME_COUNT

# 二、🛠️ 材质绑定核心函数
def force_bind_texture(obj_name, texture_path):
    obj = bpy.data.objects.get(obj_name)
    if not obj or not os.path.exists(texture_path): return
    mat = bpy.data.materials.new(name=f"Mat_{obj_name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_path)
    links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    if len(obj.data.materials) == 0: obj.data.materials.append(mat)
    else: obj.data.materials[0] = mat

def awake_vertex_color(obj_name):
    obj = bpy.data.objects.get(obj_name)
    if not obj or not obj.data.color_attributes: return
    active_attr = obj.data.color_attributes.active
    mat = bpy.data.materials.new(name=f"Mat_VC_{obj_name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    attr_node = nodes.new('ShaderNodeAttribute')
    attr_node.attribute_name = active_attr.name
    links.new(attr_node.outputs['Color'], bsdf.inputs['Base Color'])
    if len(obj.data.materials) == 0: obj.data.materials.append(mat)
    else: obj.data.materials[0] = mat

# 三、资产精准部署（采用您最新的 tissue 坐标数据）
layout_data = {
    "TABLE": {
        "path": os.path.join(DATA_DIR, "counter_surface_mesh_poisson.obj"),
        "loc": [0.0, 0.0, 0.0], "rot": [0.9948, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]
    },
    "Obj_B": {
        "path": os.path.join(DATA_DIR, "obj_b.obj"), "tex": os.path.join(DATA_DIR, "obj_b_text.jpg"), 
        "loc": [3.0103, 0.5386, 3.1579], "rot": [0.1063, 0.0618, -0.1891], "scale": [1.0, 1.0, 1.0]
    },
    "Obj_C": {
        "path": os.path.join(DATA_DIR, "obj_c.obj"), "tex": os.path.join(DATA_DIR, "obj_c_text.jpg"),
        "loc": [1.0591, 0.4646, 2.4416], "rot": [1.5707, -1.5910, 0.0], "scale": [1.0, 1.0, 1.0]
    },
    "tissue": {
        "path": os.path.join(DATA_DIR, "tissue.obj"),
        "loc": [1.2476556301116943, -0.2412528097629547, 3.766942024230957],
        "rot": [6.766481399536133, -9.467084884643555, 0.6309459209442139],
        "scale": [0.5, 0.5, 0.5]
    }
}

for item_name, info in layout_data.items():
    if not os.path.exists(info["path"]): continue
    bpy.ops.wm.obj_import(filepath=info["path"], forward_axis='Y', up_axis='Z')
    obj = bpy.context.selected_objects[0]
    obj.name = item_name
    obj.location = info["loc"]
    obj.rotation_euler = info["rot"]
    obj.scale = info["scale"]

    if item_name in ["TABLE", "tissue"]: awake_vertex_color(item_name)
    elif "tex" in info: force_bind_texture(item_name, info["tex"])

# 四、灯光与摄像机基础配置
bpy.ops.object.light_add(type='SUN', location=(-5, 5, 12))
bpy.context.object.data.energy = 6.0

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 20  
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

# 计算中心转轴与基础几何尺寸
pos_b = mathutils.Vector(layout_data["Obj_B"]["loc"])
pos_c = mathutils.Vector(layout_data["Obj_C"]["loc"])
pos_tissue = mathutils.Vector(layout_data["tissue"]["loc"])
center = (pos_b + pos_c + pos_tissue) / 3.0
start_pos = mathutils.Vector((-1.3132, -0.6302, 4.2590))

dx = start_pos.x - center.x
dy = start_pos.y - center.y
base_radius = math.sqrt(dx**2 + dy**2)
initial_angle = math.atan2(dy, dx)

# 五、🎯 核心改动 2：多段高级运动轨迹算法（慢速环绕 + 丝滑推拉镜头）
print("🎬 开始精细化编排 150 帧多阶摄像机动效关键帧...")
for f in range(1, FRAME_COUNT + 1):
    scene.frame_set(f)

    # 1. 慢速平滑旋转：150帧内均匀旋转 1.25 圈，彻底告别以前快转的感觉
    angle_offset = 2.5 * math.pi * (f - 1) / FRAME_COUNT
    current_angle = initial_angle + angle_offset

    # 2. 余弦平滑插值算法：杜绝镜头卡顿，实现完美的推拉速度渐变
    if f <= 60:
        # 第一阶段 (1~60帧)：从 1.0 倍距离平滑缩小(放大看细节)到 0.4 倍特写距离
        t = (f - 1) / 59.0
        radius_factor = 1.0 - 0.6 * (0.5 - 0.5 * math.cos(t * math.pi))
    else:
        # 第二阶段 (61~150帧)：从 0.4 倍特写距离暴增拉远到 3.5 倍巨远距离（缩小到很远）
        t = (f - 60) / 90.0
        radius_factor = 0.4 + 3.1 * (0.5 - 0.5 * math.cos(t * math.pi))

    # 计算当前帧对应的实际缩放半径
    current_radius = base_radius * radius_factor

    # 计算摄像机在当前帧的三维空间位置
    cam.location.x = center.x + current_radius * math.cos(current_angle)
    cam.location.y = center.y + current_radius * math.sin(current_angle)

    # 3. 智能无人机高度协同：拉远时镜头同步升高（因 factor 最大可达 3.5），转为大气恢弘的俯瞰视角
    cam.location.z = start_pos.z * (0.6 + 0.4 * radius_factor)

    # 4. 强行锁定：无论拉多远，镜头中轴线死死咬住三个资产的物理中心
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # 记录当前帧的关键帧数据
    cam.keyframe_insert(data_path="location", frame=f)
    cam.keyframe_insert(data_path="rotation_euler", frame=f)

# 六、视频高质量渲染导出
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'

video_output = os.path.join(OUTPUT_DIR, "object_fusion_orbit_v5.mp4")
scene.render.filepath = video_output

print("📺 正在调用后台 Cycles 引擎，渲染全新高级镜头（慢速 + 先放大特写 + 后极远拉伸）视频...")
bpy.ops.render.render(animation=True)
print("🏁 高级镜头视频管线渲染圆满落幕！")
