_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

train_dataset=dict(
    type='LVISV1Dataset',
    ann_file='data/lvis_v1/annotations/lvis_v1_train_seen.json',
    img_prefix='data/lvis_v1/',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(
            type='Resize',
            img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                       (1333, 768), (1333, 800)],
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
    ]
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=train_dataset,
    )
)

model = dict(
    type='PromptDet',
    # backbone=dict(
    #     init_cfg=dict(type='Pretrained', checkpoint='checkpoints/resnet50-19c8e357.pth')
    # ),
    roi_head=dict(
        type='PromptDetRoIHead',
        bbox_head=dict(
            type='PromptBBoxHead',
            num_classes=1203,
            reg_class_agnostic=True,
            embedding_file="promptdet_resources/lvis_manual_category_embeddings.pt",
            loss_cls=dict(
                type='MaskedCrossEntropyLoss', 
                use_sigmoid=True, 
                loss_weight=1.0,
                num_classes=1203,
                base_ind_file="promptdet_resources/lvis_base_inds.txt" 
            ),
        ),
        mask_head=dict(
            num_classes=1203, 
            class_agnostic=True
        ),
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            max_per_img=300
        )
    )
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(metric=['bbox', 'segm'], interval=12)
