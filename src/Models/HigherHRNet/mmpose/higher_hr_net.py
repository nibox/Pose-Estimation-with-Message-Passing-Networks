import torch.nn as nn
from .keypoint_head import BottomUpHigherResolutionHead
from .backbone import HRNet

class BottomUp(nn.Module):
    """Bottom-up pose detectors.
    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 backbone,
                 keypoint_head,
                 ):
        super().__init__()

        self.backbone: nn.Module = HRNet(**backbone)
        self.keypoint_head: nn.Module = BottomUpHigherResolutionHead(**keypoint_head)

    def forward(self, x):
        output = self.backbone(x)
        return self.keypoint_head(output)



def get_mmpose_hrnet(config):
    backbone, keypoint_head = translate_config(config)
    return BottomUp(backbone, keypoint_head)

def translate_config(config):
    cfg = config.MODEL.HRNET.EXTRA
    backbone = dict(
        #type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=cfg.STAGE2.NUM_MODULES,
                num_branches=cfg.STAGE2.NUM_BRANCHES,
                block=cfg.STAGE2.BLOCK,
                num_blocks=cfg.STAGE2.NUM_BLOCKS,
                num_channels=cfg.STAGE2.NUM_CHANNELS),
            stage3=dict(
                num_modules=cfg.STAGE3.NUM_MODULES,
                num_branches=cfg.STAGE3.NUM_BRANCHES,
                block=cfg.STAGE3.BLOCK,
                num_blocks=cfg.STAGE3.NUM_BLOCKS,
                num_channels=cfg.STAGE3.NUM_CHANNELS),
            stage4=dict(
                num_modules=cfg.STAGE4.NUM_MODULES,
                num_branches=cfg.STAGE4.NUM_BRANCHES,
                block=cfg.STAGE4.BLOCK,
                num_blocks=cfg.STAGE4.NUM_BLOCKS,
                num_channels=cfg.STAGE4.NUM_CHANNELS)
        ))
    in_channels = cfg.STAGE4.NUM_CHANNELS[0]
    keypoint_head = dict(
        #type='BottomUpHigherResolutionHead',
        in_channels=in_channels,
        num_joints=config.MODEL.HRNET.NUM_JOINTS,
        tag_per_joint=config.MODEL.HRNET.TAG_PER_JOINT,
        extra=dict(final_conv_kernel=1, ),
        num_deconv_layers=cfg.DECONV.NUM_DECONVS,
        num_deconv_filters=cfg.DECONV.NUM_CHANNELS,
        num_deconv_kernels=cfg.DECONV.KERNEL_SIZE,
        num_basic_blocks=cfg.DECONV.NUM_BASIC_BLOCKS,
        cat_output=cfg.DECONV.CAT_OUTPUT,
        with_ae_loss=[True, False],  # dangerours but wont change this parameter!!
        feature_fusion=config.MODEL.HRNET.FEATURE_FUSION
    )
    return backbone, keypoint_head
