import torch
from backbone_lavila.backbone_lavila import  load_backbone
from detector.detector import load_detector
import h5py
from PIL import Image 
import io 
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from torch import nn 

class Permute(nn.Module):
    """
    Permutation as an op
    """

    def __init__(self, ordering):
        super().__init__()
        self.ordering = ordering

    def forward(self, frames):
        """
        Args:
            frames in some ordering, by default (C, T, H, W)
        Returns:
            frames in the ordering that was specified
        """
        return frames.permute(self.ordering)



transform = transforms.Compose([
    Permute([1, 0, 3, 2]),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
    ])


to_tensor = transforms.PILToTensor()


class TAL_model(torch.nn.Module):

    def __init__(self, chunk_size, sampling_ratio):
        super(TAL_model, self).__init__()
        self.chunk_size = chunk_size
        self.sampling_ratio = sampling_ratio
          
        self.feature_extractor = load_backbone('/root/tesi/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth')
        self.base_detector = load_detector()

    def forward(self, video_data, feat_grad=None, stage=0):
        
        if stage == 1:  # sequentially forward the backbone
            video_feat = self.forward_stage_1(video_data)[0]
            return video_feat

        elif stage == 2:  # forward and backward the detector
            video_feat = video_data
            det_pred = self.base_detector(video_feat)
            return det_pred

        elif stage == 3:  # sequentially backward the backbone with sampled data
                self.forward_stage_3(video_data, feat_grad=feat_grad)

        elif stage == 0:  # this is for inference
                video_feat = self.forward_stage_1(video_data)
                det_pred = self.base_detector(video_feat)
                return det_pred

    
    def load_images_from_hdf5(self, input_hdf5_file):

        images_dict = {}
        with h5py.File(input_hdf5_file, 'r') as hf:
            for key in hf.keys():
                 images_dict[key] = Image.open(io.BytesIO(np.array(hf[key])))
        return images_dict
    



    def forward_stage_1(self, clips):
        # sequentially forward backbone   # clips [B, N]
        video_feat = []
        chunk_num = len(clips) // self.chunk_size
        chunks = [clips[i:i+chunk_num] for i in range(0, len(clips), chunk_num)] # these are just paths
        with torch.set_grad_enabled(False):
            for mini_frames in chunks: 
                mini_frames = self.load_images_from_hdf5(mini_frames[0])
                mini_frames = [to_tensor(img) for img in mini_frames.values()]
                mini_frames = torch.stack(mini_frames).float()
                mini_frames = transform(mini_frames)
                mini_frames = mini_frames.unsqueeze(0)
                video_feat.append(self.feature_extractor(mini_frames)) # batch_feature is N (chunks) x C (feature_dim )

        # clean cache
        video_feat = video_feat.detach()
        torch.cuda.empty_cache()
        return video_feat

    def forward_stage_3(self, video_data, feat_grad):

        B, T = video_data.shape  # batch, snippet length, 3, clip length, h, w

        # sample the snippets

        chunk_num = int(T * self.sampling_ratio / self.chunk_size + 0.99)
        assert chunk_num > 0 and chunk_num * self.chunk_size <= T

        # random sampling

        noise = torch.rand(B, T, device=video_data.device)  # noise in [0, 1]

        # sort noise for each sample

        ids_shuffle = torch.argsort(noise, dim=1)

        for chunk_idx in range(chunk_num):

            snippet_idx = ids_shuffle[:, chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size]

            # selects specific chunks of video data along the time dimension according to the indices specified in snippet_idx

            video_data_chunk = torch.gather(
                video_data,
                dim=1,
                index=snippet_idx.view(B, self.chunk_size),
            )
            feat_grad_chunk = torch.gather(
                feat_grad,
                dim=2,
                index=snippet_idx.view(B, 1, self.chunk_size).repeat(1, feat_grad.shape[1], 1),
            )
            self.feature_extractor = self.feature_extractor.train()
            with torch.set_grad_enabled(True):
                video_feat_chunk = []
                for frames in video_data_chunk:
                    frames = self.load_images_from_hdf5(frames)
                    video_feat_chunk.append(self.feature_extractor(video_data_chunk))
            assert video_feat_chunk.shape == feat_grad_chunk.shape

            # accumulate grad
            video_feat_chunk.backward(gradient=feat_grad_chunk)
