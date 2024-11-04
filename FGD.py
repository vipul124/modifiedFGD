import json
import torch
import numpy as np
import taichi as ti

@ti.data_oriented
class bilateralMatrix():
    def __init__(self):
        ti.init(arch=ti.gpu, default_fp=ti.f64)
        self.EPSILON = 1e-5

    @ti.func
    def nGaussExp(self, x: ti.f64, sigma: ti.f64):
        # Special case if sigma is 0. In this case we should have a dirac delta.
        value = 0.0
        if(sigma < self.EPSILON):
            if(x < self.EPSILON):
                value = 1.0
        else:
            value = -(x**2)/(2 * sigma**2)
        return value

    @ti.kernel
    def taichiCrossBilateralMatrix4D(self, im: ti.types.ndarray(dtype=ti.f64, ndim=3), sigma: ti.math.vec3, cbmat: ti.types.ndarray()):
        height = im.shape[0]
        width = im.shape[1]

        for py in range(height):
            for px in range(width):
                pval = ti.math.vec4(im[py,px,0], im[py,px,1], im[py,px,2], im[py,px,3])
                w = 0.0
                row = py*width+px
                for qy in range(height):
                    for qx in range(width):
                        ge = self.nGaussExp(qx - px,sigma[1])+self.nGaussExp(qy - py,sigma[0])\
                            +self.nGaussExp(im[qy,qx,0]-pval[0], sigma[2])+self.nGaussExp(im[qy,qx,1]-pval[1], sigma[2])\
                            +self.nGaussExp(im[qy,qx,2]-pval[2], sigma[2])+self.nGaussExp(im[qy,qx,1]-pval[1], sigma[2])
                        g = ti.exp(ge)
                        col = qy*width+qx
                        cbmat[row,col] = g
                        w += g
                
                for col in range(width*height):
                    cbmat[row, col]=cbmat[row, col]*(1.0/w)

    def getCrossBilateralMatrix4D(self, image, sigmas):
        height=image.shape[0]
        width = image.shape[1]

        cbmat = np.zeros([width*height,width*height])
        self.taichiCrossBilateralMatrix4D(np.ascontiguousarray(image), sigmas, cbmat)
        return cbmat


class fgdFilter():
    def __init__(self, diffusionModel, guide_image, detail=1.2, sigmas=[3,3,0.3], t_end=15, norm_steps=0):
        self.guide_image = guide_image
        self.detail = detail
        self.t_end = t_end
        self.sigmas = sigmas
        self.norm_steps = norm_steps
        self.model = diffusionModel
        self.bilateral_matrix_4d = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.guide_latent = None
        self.guide_structure = None
        self.guide_structure_normalized = None

        self.init_guide_latent = None
        self.init_guide_structure = None
        self.init_guide_stucture_normalized=None
        
        self.init_bilateral_matrix_4d=None
        self.set_guide_image(guide_image)

    def set_ST(self, detail=1.6, recompute_matrix=True, sigmas=[3,3,0.3], t_end=15, norm_steps=50):
        if recompute_matrix:
            self.set_bilateral_matrix(sigmas)
        self.detail = detail
        self.t_end = t_end
        self.norm_steps = norm_steps

    def reset(self):
        self.init_guide_latent = self.guide_latent
        self.guide_structure = self.init_guide_structure
        self.guide_structure_normalized = self.init_guide_structure_normalized
        self.bilateral_matrix_4d = self.init_bilateral_matrix_4d

    def set_guide_image(self, guide_image):
        self.guide_latent = self.model.encode_image(guide_image)
        self.guide_image = guide_image
        if self.sigmas != None:
            self.set_bilateral_matrix(self.sigmas)

    def set_bilateral_matrix(self,sigmas):
        assert len(sigmas)==2 or len(sigmas)==3, "sigmas has invalid number of entries (either 2 or 3)"
        sigmas = np.array(sigmas).astype(np.double)
        if len(sigmas) == 2:
            sigmas = np.insert(sigmas, 1, sigmas[0])

        guide_latent_processed = self.guide_latent.detach().cpu().permute(0, 2, 3, 1).numpy()
        guide_latent_processed = np.squeeze(guide_latent_processed)
        bilateral_matrix = bilateralMatrix().getCrossBilateralMatrix4D(guide_latent_processed.astype('double'),sigmas)
        self.bilateral_matrix_4d = torch.Tensor(bilateral_matrix).unsqueeze(0).repeat((4,1,1)).to(self.device)
        guide_structure_latent = torch.matmul(self.bilateral_matrix_4d, self.guide_latent.reshape(4,4096,1))
        guide_structure_latent = guide_structure_latent.reshape(1,4,64,64)

        guide_mean = torch.mean(guide_structure_latent, (2,3), keepdim=True)
        guide_std = torch.std(guide_structure_latent, (2,3), keepdim=True)

        self.guide_structure_normalized = (guide_structure_latent - guide_mean) / guide_std
        self.guide_structure = guide_structure_latent

        self.init_guide_structure = self.guide_structure
        self.init_guide_structure_normalized=self.guide_structure_normalized
        self.init_bilateral_matrix_4d = self.bilateral_matrix_4d

        self.sigmas = sigmas.tolist()
    
    def get_residual_structure(self, latents):
        current_structure = torch.matmul(self.bilateral_matrix_4d, latents.reshape(4,4096,1))
        current_structure = current_structure.reshape(1,4,64,64)

        d_structure = self.guide_structure - current_structure
        return d_structure
    
    def get_structure(self, latents, bm_4d=None):
        if bm_4d ==None:
            bm_4d = self.bilateral_matrix_4d
        structure = torch.matmul(bm_4d, latents.reshape(4,4096,1))
        structure = structure.reshape(1,4,64,64)
        return structure

    def get_guidance(self, latents, input_latents, scheduler, t):
        guide_low = self.guide_structure
        
        st_low = self.get_structure(latents)
        st_high = latents - st_low

        weight= self.detail
        d = guide_low - st_low
        return weight, d

    
    def get_guidance_normalized(self, latents, input_latents, scheduler, t):
        current_structure = self.get_structure(latents)
        guide_structure = self.guide_structure
            
        current_mean = torch.mean(current_structure, (2,3), keepdim=True)
        current_std = torch.std(current_structure, (2,3), keepdim=True)

        guide_structure_renormalized = self.guide_structure_normalized * current_std + current_mean
        d_structure_renormalized = guide_structure_renormalized - current_structure

        residual_score = torch.mean(torch.abs(d_structure_renormalized)) 
        weight = self.detail
        return weight, d_structure_renormalized
       
    def get_params(self):
        params = {
            'guide image':self.guide_image,
            'detail':self.detail,
            'sigmas':self.sigmas,
            't_end':self.t_end,
            'norm steps':self.norm_steps,
        }
        return params
    
    def __str__(self):
        return (json.dumps(self.get_params(), indent=2))
