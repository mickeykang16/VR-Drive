import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(novel_FovX, 
           novel_FovY, 
           novel_height, 
           novel_width, 
           novel_world_view_transform, 
           novel_full_proj_transform, 
           novel_camera_center, 
           pts_xyz, 
           pts_rgb, 
           rotations, 
           scales, 
           opacity, 
           shs, 
           bg_color):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    bg_color = torch.tensor(bg_color, dtype=torch.float32).cuda()
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True).cuda()
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(novel_FovX * 0.5)
    tanfovy = math.tan(novel_FovY * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(novel_height),
        image_width=int(novel_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=novel_world_view_transform,
        projmatrix=novel_full_proj_transform,
        sh_degree=3,
        campos=novel_camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, _, _ = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=pts_rgb,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return rendered_image
