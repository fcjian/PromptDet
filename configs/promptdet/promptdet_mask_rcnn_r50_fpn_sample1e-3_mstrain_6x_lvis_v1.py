_base_ = './promptdet_mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[48, 66])
runner = dict(type='EpochBasedRunner', max_epochs=72)
evaluation = dict(metric=['bbox', 'segm'], interval=12)

