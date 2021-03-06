###############################################################################
# 1. Set variables

# checkpoint 저장용 workdir
root_workdir = "workdir"
# GPU당 batch 크기
samples_per_gpu = 16
# 이미지 크기 
size = (48, 192)
# 이미지 평균, 표준편차
mean, std = 0.5, 0.5

# 토큰
character = ""
# case sensitive 여부
sensitive = False
# 최대 시퀀스 길이
batch_max_length = 300

# normalization 설정
norm_cfg = dict(type="BN")
# 클래스 개수 = 241(토큰) + 스페셜 토큰
num_class = 244
# 기본 채널수
base_channel = 16

###############################################################################
# 2. Set model config
inference = dict(
    transform=[
        dict(type="Sensitive", sensitive=sensitive),
        # dict(type="Filter", need_character=character),
        dict(type="ToGray"),
        dict(type="Resize", size=size),
        dict(type="Normalize", mean=mean, std=std),
        dict(type="ToTensor"),
    ],
    converter=dict(
        type="CustomConverter", character=character, batch_max_length=batch_max_length,
    ),
    model=dict(
        type="GModel",
        need_text=False,
        body=dict(
            type="GBody",
            pipelines=[
                dict(
                    type="FeatureExtractorComponent",
                    from_layer="input",
                    to_layer="cnn_feat",
                    arch=dict(
                        encoder=dict(
                            backbone=dict(
                                type="GBackbone",
                                layers=[
                                    dict(
                                        type="ConvModule",
                                        in_channels=1,
                                        out_channels=32 + base_channel,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        norm_cfg=norm_cfg,
                                    ),  # 48, 192
                                    dict(
                                        type="ConvModule",
                                        in_channels=32 + base_channel,
                                        out_channels=64 + base_channel * 2,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        norm_cfg=norm_cfg,
                                    ),  # 48, 192 # c0
                                    dict(
                                        type="MaxPool2d",
                                        kernel_size=2,
                                        stride=2,
                                        padding=0,
                                    ),  # 24, 96
                                    dict(
                                        type="BasicBlocks",
                                        inplanes=64 + base_channel * 2,
                                        planes=128 + base_channel * 4,
                                        blocks=1,
                                        stride=1,
                                        norm_cfg=norm_cfg,
                                        plug_cfg=dict(
                                            type="CBAM",
                                            gate_channels=128 + base_channel * 4,
                                            reduction_ratio=16,
                                            norm_cfg=dict(
                                                type=norm_cfg["type"], momentum=0.01
                                            ),
                                        ),
                                    ),
                                    # 24, 96
                                    dict(
                                        type="ConvModule",
                                        in_channels=128 + base_channel * 4,
                                        out_channels=128 + base_channel * 4,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        norm_cfg=norm_cfg,
                                    ),  # 24, 96
                                    dict(
                                        type="NonLocal2d",
                                        in_channels=128 + base_channel * 4,
                                        sub_sample=True,
                                    ),  # c1
                                    dict(
                                        type="MaxPool2d",
                                        kernel_size=2,
                                        stride=2,
                                        padding=0,
                                    ),  # 12, 48
                                    dict(
                                        type="BasicBlocks",
                                        inplanes=128 + base_channel * 4,
                                        planes=256 + base_channel * 8,
                                        blocks=4,
                                        stride=1,
                                        norm_cfg=norm_cfg,
                                        plug_cfg=dict(
                                            type="CBAM",
                                            gate_channels=256 + base_channel * 8,
                                            reduction_ratio=16,
                                            norm_cfg=dict(
                                                type=norm_cfg["type"], momentum=0.01
                                            ),
                                        ),
                                    ),
                                    # 12, 48
                                    dict(
                                        type="ConvModule",
                                        in_channels=256 + base_channel * 8,
                                        out_channels=256 + base_channel * 8,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        norm_cfg=norm_cfg,
                                    ),  # 12, 48
                                    dict(
                                        type="NonLocal2d",
                                        in_channels=256 + base_channel * 8,
                                        sub_sample=True,
                                    ),  # c2
                                    dict(
                                        type="MaxPool2d", kernel_size=2, stride=(2, 2)
                                    ),
                                    # 6, 24
                                    dict(
                                        type="BasicBlocks",
                                        inplanes=256 + base_channel * 8,
                                        planes=512 + base_channel * 16,
                                        blocks=7,
                                        stride=1,
                                        norm_cfg=norm_cfg,
                                        plug_cfg=dict(
                                            type="CBAM",
                                            gate_channels=512 + base_channel * 16,
                                            reduction_ratio=16,
                                            norm_cfg=dict(
                                                type=norm_cfg["type"], momentum=0.01
                                            ),
                                        ),
                                    ),
                                    # 6, 24
                                    dict(
                                        type="ConvModule",
                                        in_channels=512 + base_channel * 16,
                                        out_channels=512 + base_channel * 16,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        norm_cfg=norm_cfg,
                                    ),  # 6, 24
                                    dict(
                                        type="BasicBlocks",
                                        inplanes=512 + base_channel * 16,
                                        planes=512 + base_channel * 16,
                                        blocks=5,
                                        stride=1,
                                        norm_cfg=norm_cfg,
                                        plug_cfg=dict(
                                            type="CBAM",
                                            gate_channels=512 + base_channel * 16,
                                            reduction_ratio=16,
                                            norm_cfg=dict(
                                                type=norm_cfg["type"], momentum=0.01
                                            ),
                                        ),
                                    ),
                                    # 6, 24
                                    dict(
                                        type="NonLocal2d",
                                        in_channels=512 + base_channel * 16,
                                        sub_sample=True,
                                    ),  # c3
                                    dict(
                                        type="ConvModule",
                                        in_channels=512 + base_channel * 16,
                                        out_channels=512 + base_channel * 16,
                                        kernel_size=2,
                                        stride=(2, 1),
                                        norm_cfg=norm_cfg,
                                    ),  # 3, 23
                                    dict(
                                        type="BasicBlocks",
                                        inplanes=512 + base_channel * 16,
                                        planes=512 + base_channel * 16,
                                        blocks=3,
                                        stride=1,
                                        norm_cfg=norm_cfg,
                                        plug_cfg=dict(
                                            type="CBAM",
                                            gate_channels=512 + base_channel * 16,
                                            reduction_ratio=16,
                                            norm_cfg=dict(
                                                type=norm_cfg["type"], momentum=0.01
                                            ),
                                        ),
                                    ),
                                    dict(
                                        type="ConvModule",
                                        in_channels=512 + base_channel * 16,
                                        out_channels=512 + base_channel * 16,
                                        kernel_size=2,
                                        stride=1,
                                        padding=0,
                                        norm_cfg=norm_cfg,
                                    ),  # 2, 24  # c4
                                ],
                            ),
                        ),
                        decoder=dict(
                            type="GFPN",
                            neck=[
                                dict(
                                    type="JunctionBlock",
                                    top_down=None,
                                    lateral=dict(
                                        from_layer="c4",
                                        type="ConvModule",
                                        in_channels=512 + base_channel * 16,
                                        out_channels=512,
                                        kernel_size=1,
                                        activation="relu",
                                        norm_cfg=norm_cfg,
                                    ),
                                    post=None,
                                    to_layer="p5",
                                ),  # 32
                                # model/decoder/blocks/block2
                                dict(
                                    type="JunctionBlock",
                                    fusion_method="add",
                                    top_down=dict(
                                        from_layer="p5",
                                        upsample=dict(
                                            type="Upsample",
                                            size=(6, 24),
                                            scale_bias=0,
                                            mode="bilinear",
                                            align_corners=True,
                                        ),
                                    ),
                                    lateral=dict(
                                        from_layer="c3",
                                        type="ConvModule",
                                        in_channels=512 + base_channel * 16,
                                        out_channels=512,
                                        kernel_size=1,
                                        activation="relu",
                                        norm_cfg=norm_cfg,
                                    ),
                                    post=None,
                                    to_layer="p4",
                                ),  # 16
                                # model/decoder/blocks/block3
                                dict(
                                    type="JunctionBlock",
                                    fusion_method="add",
                                    top_down=dict(
                                        from_layer="p4",
                                        upsample=dict(
                                            type="Upsample",
                                            size=(12, 48),
                                            scale_bias=0,
                                            mode="bilinear",
                                            align_corners=True,
                                        ),
                                    ),
                                    lateral=dict(
                                        from_layer="c2",
                                        type="ConvModule",
                                        in_channels=256 + base_channel * 8,
                                        out_channels=512,
                                        kernel_size=1,
                                        activation="relu",
                                        norm_cfg=norm_cfg,
                                    ),
                                    post=None,
                                    to_layer="p3",
                                ),  # 8
                                # fusion the features
                                dict(
                                    type="JunctionBlock",
                                    fusion_method=None,
                                    top_down=dict(
                                        from_layer="p5",
                                        trans=dict(
                                            type="ConvModule",
                                            in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            padding=1,
                                            conv_cfg=dict(type="Conv"),
                                            norm_cfg=norm_cfg,
                                            activation="relu",
                                            inplace=True,
                                        ),
                                        upsample=dict(
                                            type="Upsample",
                                            size=(6, 24),
                                            scale_bias=0,
                                            mode="bilinear",
                                            align_corners=True,
                                        ),
                                    ),
                                    lateral=None,
                                    post=None,
                                    to_layer="p5_1",
                                ),  # 6, 24
                                dict(
                                    type="JunctionBlock",
                                    fusion_method=None,
                                    top_down=dict(
                                        from_layer="p5_1",
                                        trans=dict(
                                            type="ConvModule",
                                            in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            padding=1,
                                            conv_cfg=dict(type="Conv"),
                                            norm_cfg=norm_cfg,
                                            activation="relu",
                                            inplace=True,
                                        ),
                                        upsample=dict(
                                            type="Upsample",
                                            size=(12, 48),
                                            scale_bias=0,
                                            mode="bilinear",
                                            align_corners=True,
                                        ),
                                    ),
                                    lateral=None,
                                    post=None,
                                    to_layer="p5_2",
                                ),  # 12, 48
                                dict(
                                    type="JunctionBlock",
                                    fusion_method="add",
                                    top_down=dict(
                                        from_layer="p4",
                                        trans=dict(
                                            type="ConvModule",
                                            in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            padding=1,
                                            conv_cfg=dict(type="Conv"),
                                            norm_cfg=norm_cfg,
                                            activation="relu",
                                            inplace=True,
                                        ),
                                        upsample=dict(
                                            type="Upsample",
                                            size=(12, 48),
                                            scale_bias=0,
                                            mode="bilinear",
                                            align_corners=True,
                                        ),
                                    ),
                                    lateral=dict(from_layer="p5_2",),
                                    post=None,
                                    to_layer="p4_1",
                                ),  # 12, 48
                                dict(
                                    type="JunctionBlock",
                                    fusion_method="add",
                                    top_down=dict(
                                        from_layer="p3",
                                        trans=dict(
                                            type="ConvModule",
                                            in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            padding=1,
                                            conv_cfg=dict(type="Conv"),
                                            norm_cfg=norm_cfg,
                                            activation="relu",
                                            inplace=True,
                                        ),
                                    ),
                                    lateral=dict(from_layer="p4_1",),
                                    post=None,
                                    to_layer="p3_1",
                                ),  # 12, 48
                            ],
                        ),
                        collect=dict(type="CollectBlock", from_layer="p3_1"),
                    ),
                ),
                dict(
                    type="FeatureExtractorComponent",
                    from_layer="cnn_feat",
                    to_layer="non_local_feat",
                    arch=dict(
                        encoder=dict(
                            backbone=dict(
                                type="GBackbone",
                                layers=[
                                    dict(
                                        type="NonLocal2d",
                                        in_channels=512,
                                        sub_sample=True,
                                    ),  # c0
                                ],
                            ),
                        ),
                        collect=dict(type="CollectBlock", from_layer="c0"),
                    ),
                ),
            ],
        ),
        head=dict(
            type="MultiHead",
            in_channels=512,
            num_class=num_class,
            from_layer="non_local_feat",
            batch_max_length=batch_max_length,
            pool=dict(type="AdaptiveAvgPool2d", output_size=1,),
        ),
    ),
)
###############################################################################
# 3.Set logger, metric, cuda settings
common = dict(
    seed=1111,
    logger=dict(
        handlers=(
            dict(type="StreamHandler", level="INFO"),
            dict(type="FileHandler", level="INFO"),
        ),
    ),
    cudnn_deterministic=False,
    cudnn_benchmark=True,
    metric=dict(type="Accuracy"),
    dist_params=dict(backend="nccl"),
)
###############################################################################
# 4. Define file directory

# 루트 디렉터리
data_root = "/opt/ml/input/data/"

# 학습 데이터셋 디렉터리
train_root = data_root + "train_dataset/"
valid_root = data_root + "train_dataset/"

# 테스트 데이터셋 디렉터리
test_root = data_root + "eval_dataset/"
###############################################################################
# 5. Set dataset parameters
# train dataset
dataset_params = dict(
    batch_max_length=batch_max_length, 
    data_filter=False, 
    character=character, 
    root=train_root+"images/",
    gt_txt=train_root+"custom_train.txt",
)

# valid dataset
valid_dataset_params = dict(
    batch_max_length=batch_max_length, 
    data_filter=False, 
    character=character,
    root=train_root+"images/",
    gt_txt=train_root+"custom_test.txt"
)

# test dataset
test_dataset_params = dict(
    batch_max_length=batch_max_length, 
    data_filter=False, 
    character=character,
    root=test_root+"images/",
    gt_txt=test_root+"input.txt"
)
###############################################################################
# 6. Create Dataset

train_dataset = dict(type="TxtDataset", **dataset_params)
valid_dataset = dict(type="TxtDataset", **valid_dataset_params)
test_dataset = dict(type="TxtDataset", **test_dataset_params)

###############################################################################
# 7. Set test configs
test = dict(
    data=dict(
        dataloader=dict(
            type="DataLoader",
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=4,
            shuffle=False,
        ),
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=test_dataset,
        transform=inference["transform"],
    ),
    postprocess_cfg=dict(sensitive=sensitive, character=character,),
)

###############################################################################
# 8. Set train configs

# 학습 iterations 수
max_iterations = 420000
# lr 변화 지점 
milestones = [150000, 250000]

# train transforms
train_transforms = [
    dict(type="Sensitive", sensitive=sensitive),
    # dict(type="Filter", need_character=character),
    dict(type="MotionBlur", p=0.5),
    dict(type="ColorJitter", p=0.5),
    dict(type="ToGray"),
    dict(type="Resize", size=size),
    dict(type="GaussNoise", p=0.5),
    dict(type="Normalize", mean=mean, std=std),
    dict(type="ToTensor"),
]

train = dict(
    data=dict(
        train=dict(
            dataloader=dict(
                type="DataLoader", samples_per_gpu=samples_per_gpu, workers_per_gpu=4,
            ),
            # sampler=dict(
            #     type='BalanceSampler',
            #     samples_per_gpu=samples_per_gpu,
            #     shuffle=True,
            #     oversample=True,
            #     seed=common['seed'],  # if not set, default seed is 0.
            # ),
            dataset=train_dataset,
            transform=train_transforms,
        ),
        val=dict(
            dataloader=dict(
                type="DataLoader",
                samples_per_gpu=samples_per_gpu,
                workers_per_gpu=4,
                shuffle=False,
            ),
            dataset=valid_dataset,
            transform=inference["transform"],
        ),
    ),
    optimizer=dict(type="Adadelta", lr=1.0, rho=0.95, eps=1e-8),
    criterion=dict(type="LabelSmoothingCrossEntropy"),
    lr_scheduler=dict(
        type="StepLR", iter_based=True, milestones=milestones, warmup_epochs=0.2,
    ),
    max_iterations=max_iterations,
    log_interval=10,
    trainval_ratio=2000,
    snapshot_interval=20000,
    save_best=True,
    resume=None,
)
