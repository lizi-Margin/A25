import torch, cv2, numpy as np
import third_party.DehazeDDPM.core.metrics as Metrics
import third_party.DehazeDDPM.model as Model
from siri_utils.preprocess import _post_compute

def get_ddpm_model():
    from json import load
    with open("./ddpm_dense.jsonc", 'r') as f:
        opt = load(f)

    diffusion = Model.create_model(opt)
    return diffusion

class ddpmWrapper():
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self.index = 0

    def __call__(self, *args, **kwds):
        return self.act(*args, **kwds)

    @torch.no_grad()
    def act(self, wl_images):
        cv2.imshow("wl_img", cv2.cvtColor(_post_compute(wl_images)[0], cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        data = {} 
        data['SR'] = wl_images
        data['HR'] = wl_images  # no use
        data['Index'] = torch.Tensor(self.index)
        self.diffusion.feed_data(data)
        self.diffusion.test(continous=False)
        visuals = self.diffusion.get_current_visuals()
        out_img = Metrics.tensor2img(visuals['Out'])  # uint8
        # hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        # lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
        cv2.imshow("out_img", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        # cv2.imshow("hr_img", cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR))
        # cv2.imshow("lr_img", cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.index += 1
        return np.array([out_img])
    
    def eval(self):
        # auto set, no use
        return self