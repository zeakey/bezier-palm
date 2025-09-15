import mano
import torch
import numpy as np

from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        SoftPhongShader,
        MeshRasterizer,
        PointLights)


def get_mano(batch_size, is_right=True):
    if is_right:
        mano_model = 'assets/mano_models/MANO_RIGHT.pkl'
    else:
        mano_model = 'assets/mano_models/MANO_LEFT.pkl'
    return mano.load(
            model_path=mano_model,
            is_right= is_right,
            num_pca_comps=45,
            batch_size=batch_size,
            flat_hand_mean=True)

def get_vjf(model, betas, pose, global_orient, translate, mode='right', normalize=True):
    """
    get vertices, joints and faces
    return:
        vertices: (n, 778, 3)
        joints: (n, 21, 3)
        faces: 
    """
    batch_size = betas.size(0)
    assert pose.size(0) == global_orient.size(0) == translate.size(0)
    
    output = model(betas=betas,
                  global_orient=global_orient,
                  hand_pose=pose,
                  transl=translate,
                  return_verts=True,
                  return_tips = True)
    verts = output.vertices
    joints = output.joints
    faces = torch.tensor(model.faces.astype(int)).unsqueeze(dim=0).repeat(batch_size, 1, 1)
    nv = verts.size(1)
    nj = joints.size(1)
    if normalize:
        mean = verts.mean(dim=1, keepdim=True)
        verts -= mean
        max = verts.max(dim=1, keepdim=True).values
        verts /= max

        joints -= mean
        joints /= max
    return verts, joints, faces


def render(mesh, R, T, image_size=(448, 448), light=[0, 0, 0]):
    device = R.device
    cameras = FoVPerspectiveCameras(
        R=R, T=T, device=device
    )

    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    lights = PointLights(device=device, location=[light])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(mesh)
    return images, cameras
