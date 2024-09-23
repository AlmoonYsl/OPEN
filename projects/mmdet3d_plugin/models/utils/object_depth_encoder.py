import copy

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmcv.cnn import xavier_init, constant_init, build_norm_layer
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import FFN, build_attention
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.runner import ModuleList
from mmcv.runner.base_module import BaseModule


class DeformableDetrTransformerEncoder(BaseModule):
    def __init__(self,
                 num_layers,
                 layer_cfg,
                 num_classes=10,
                 position_embeding=None,
                 init_cfg=None) -> None:

        super().__init__(init_cfg=init_cfg)
        assert num_layers == 1, 'ODE is one layer'
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.layer_cfg = layer_cfg
        self.position_embeding = position_embeding
        self._init_layers()

    def _init_layers(self) -> None:
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

        depth_branch = []
        for _ in range(2):
            depth_branch.append(Linear(self.embed_dims, self.embed_dims))
            depth_branch.append(nn.LayerNorm(self.embed_dims))
            depth_branch.append(nn.ReLU(inplace=True))
        depth_branch.append(Linear(self.embed_dims, 1))
        depth_branch = nn.Sequential(*depth_branch)
        self.depth_branch = nn.ModuleList([copy.deepcopy(depth_branch) for _ in range(self.num_layers)])

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(self, query, query_pos, value,
                key_padding_mask, spatial_shapes,
                level_start_index, reference_points, reference_depth_map, reference_location, intrinsics, img_metas,
                **kwargs):
        intermediate = list()
        intermediate_depth_map = [reference_depth_map]

        for i, layer in enumerate(self.layers):
            reference_points, query_pos = self.position_embeding(reference_location, reference_depth_map, intrinsics,
                                                                 img_metas)
            bn, h, w, _ = query_pos.size()
            query_pos = query_pos.reshape(bn, h*w, -1)
            reference_points = reference_points.reshape(bn, h*w, 3)

            query = layer(
                query=query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points.detach(),
                intrinsics=intrinsics,
                img_metas=img_metas,
                **kwargs)

            intermediate.append(query)

            res_depth = self.depth_branch[i](query)
            res_depth = res_depth + intermediate_depth_map[i]

            reference_depth_map = res_depth
            intermediate_depth_map.append(res_depth)


        intermediate = torch.stack(intermediate)
        intermediate_depth_map = torch.stack(intermediate_depth_map[1:])

        return intermediate, intermediate_depth_map


class DeformableDetrTransformerEncoderLayer(BaseModule):
    def __init__(self,
                 attn_cfg=dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg=dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.attn_cfg = attn_cfg
        if 'batch_first' not in self.attn_cfg:
            self.attn_cfg['batch_first'] = True
        else:
            assert self.attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        self.attn = build_attention(self.attn_cfg)
        self.embed_dims = self.attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self, query, query_pos, value, key_padding_mask, **kwargs):
        query = self.attn(
            query=query,
            value=value,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)
        return query


@ATTENTION.register_module()
class DeformableFeatureAggregation3D(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_heads=8,
            num_levels=4,
            num_cams=6,
            dropout=0.1,
            num_pts=13,
            im2col_step=64,
            batch_first=True,
            bias=2.,
    ):
        super(DeformableFeatureAggregation3D, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.group_dims = (self.embed_dims // self.num_heads)
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.weights_fc = nn.Linear(self.embed_dims, self.num_heads * self.num_levels * num_pts)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.sampling_offset = nn.Linear(self.embed_dims, num_pts * 3)
        self.cam_embed = nn.Sequential(
            nn.Linear(12, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),
        )
        self.drop = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        self.bias = bias
        self.min_depth = 1e-5
        self.max_depth = 61.2

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        nn.init.uniform_(self.sampling_offset.bias.data, -self.bias, self.bias)

    def get_camera_pos(self, points, pc_range):
        points = points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
        return points

    def forward(self, query, value, query_pos, spatial_shapes, level_start_index,
                reference_points, intrinsics, img_metas, pc_range, key_padding_mask=None):

        bs, num_query = reference_points.shape[:2]
        intrinsics = intrinsics.reshape(bs, 1, 4, 4)
        cam2img = intrinsics
        reference_points = self.get_camera_pos(reference_points, pc_range)

        if query_pos is not None:
            key_points = reference_points.unsqueeze(-2) + self.sampling_offset(query + query_pos).reshape(bs, num_query, -1, 3)
        else:
            key_points = reference_points.unsqueeze(-2) + self.sampling_offset(query).reshape(bs, num_query, -1, 3)

        weights = self._get_weights(query, query_pos, cam2img)
        features = self.feature_sampling(value, spatial_shapes, level_start_index, key_points, weights,
                                         cam2img, img_metas)
    
        output = self.output_proj(features)
        output = self.drop(output) + query
        return output

    def _get_weights(self, query, query_pos, cam2img):
        bs, num_anchor = query.shape[:2]
        cam2img = cam2img[..., :3, :].reshape(bs, -1)
        cam_embed = cam2img / 1e3
        cam_embed = self.cam_embed(cam_embed)  # B, N, C
        if query_pos is not None:
            feat_pos = (query + query_pos) + cam_embed.unsqueeze(1)
        else:
            feat_pos = query + cam_embed.unsqueeze(1)
        weights = self.weights_fc(feat_pos).reshape(bs, num_anchor, -1, self.num_heads).softmax(dim=-2)
        weights = weights.reshape(bs, num_anchor, -1, self.num_heads).permute(0, 1, 3, 2).contiguous()
        return weights

    def feature_sampling(self, feat_flatten, spatial_flatten, level_start_index, key_points, weights, cam2img,
                         img_metas):
        bs, num_query, _ = key_points.shape[:3]

        pts_extand = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1).unsqueeze(dim=-1)
        points_2d = torch.matmul(cam2img.unsqueeze(dim=2), pts_extand).squeeze(-1)[..., :3]

        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        points_2d[..., 0:1] = points_2d[..., 0:1] / img_metas[0]['pad_shape'][0][1]
        points_2d[..., 1:2] = points_2d[..., 1:2] / img_metas[0]['pad_shape'][0][0]

        points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_heads, self.num_levels, 1, 1)

        bn, num_value, _ = feat_flatten.size()
        feat_flatten = feat_flatten.reshape(bn, num_value, self.num_heads, -1)
        weights = weights.view(bs, num_query, self.num_heads, self.num_levels, -1)

        output = MultiScaleDeformableAttnFunction.apply(
            feat_flatten, spatial_flatten, level_start_index, points_2d,
            weights, self.im2col_step)

        output = output.reshape(bs, num_query, -1)

        return output