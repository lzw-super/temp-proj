# LingBot-Map 论文训练数据：室内公开数据集整理

更新时间：2026-06-17

论文来源：[paper-pdf/Geometric Context Transformer for.pdf](paper-pdf/Geometric%20Context%20Transformer%20for.pdf)，主要依据 Sec. 4.3、Table 1 和 Appendix A.5。

## 筛选口径

本文只整理 LingBot-Map 训练语料中与“室内环境”相关的公开数据集。真实采集、合成、虚拟、渲染数据都纳入；但只关注室内环境部分，不分析纯室外、自动驾驶、航拍、对象中心、纯物体 mesh 或内部私有数据。

对 TartanAir、TartanGround、DL3DV 这类同时包含室内和室外的数据集，本文单独放在“可筛室内子集”部分；论文 Table 1 的 stage 采样比例是对整个数据集的比例，论文没有给出室内子集单独比例。

另外，7-Scenes、NRGBD、ETH3D 等在论文 Sec. 5 中主要作为 evaluation benchmark 出现，不属于 Table 1 的训练数据清单，因此不纳入主表。

## 论文中的 Stage 定义

- Stage 1 / Base Model Training：离线基础模型训练；每次从场景中采样 2 到 24 帧，使用全局 attention，不要求时间顺序，目标是学习通用几何先验。
- Stage 2 / Streaming Model Training：流式模型训练；初始化自 Stage 1，将全局 attention 替换为 GCA，并把训练视角数从 24 逐步增加到 320，更依赖长轨迹视频数据。

## 明确室内环境数据集

| 数据集 | 类型 | 获取途径 | 规模 / 大小 | 论文中用途 | 备注 |
|---|---|---|---:|---|---|
| HyperSim | 合成/虚拟室内，多视角 | Apple/GitHub：[apple/ml-hypersim](https://github.com/apple/ml-hypersim)，项目页：[Hypersim](https://machinelearning.apple.com/research/hypersim) | 461 个室内场景，论文原始规模 77,400 张图；公开 release 为 74,619 张图，约 1.9TB | Stage 1：5.8%；Stage 2：5.7% | 室内合成数据，提供几何、相机、语义等 dense GT；论文 Appendix A.1 提到其深度为米单位 float `.npy`。 |
| SceneNet RGB-D / SceneRGBD | 合成/虚拟室内，RGB-D video | 官方页：[SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html) | 5M rendered RGB-D images，15K+ synthetic indoor trajectories；训练集 263GB，验证集 15GB | Stage 1：7.3%；Stage 2：5.7% | 论文表格写作 `SceneRGBD [43]`，对应引用是 SceneNet RGB-D。 |
| Aria Synthetic Environments (ASE) | 合成/虚拟室内，egocentric video | Project Aria：[ASE dataset](https://www.projectaria.com/datasets/ase)，文档：[Project Aria ASE docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset)，下载：[ASE download](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_download_dataset) | 100K unique multi-room interior scenes / sequences，58M+ images，约 23TB | Stage 1：7.3%；Stage 2：5.7% | 程序化 apartment/interior layout，模拟 Aria 眼镜传感器和约 2 分钟室内轨迹。 |
| Replica | 真实室内重建 mesh，可渲染/仿真 | GitHub：[facebookresearch/replica-dataset](https://github.com/facebookresearch/replica-dataset)，论文：[arXiv](https://arxiv.org/abs/1906.05797) | 18 个高质量室内 3D scene reconstruction | Stage 1：Sec. 4.3 正文列入；Table 1 未给比例；Stage 2：未见表格比例 | 论文正文把 Replica 列在 Stage 1 video datasets 中，但 Table 1 没有单独行；复现采样比例时需要自行决定权重。 |
| Aria Digital Twin (ADT) | 真实室内 egocentric 多传感器 | Project Aria：[ADT dataset](https://www.projectaria.com/datasets/adt)，下载文档：[ADT download](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/dataset_download) | 当前官方文档：236 sequences，采集于 apartment 和 single-room office 两个真实室内空间；不含 MPS 约 3.5TB | Stage 1：1.0%；Stage 2：未使用 | ADT 同时包含真实 sensor/GT 与部分合成/渲染派生内容；按本口径都可记录，但训练时建议明确所选 data type。 |
| ScanNet++ | 真实室内 RGB-D / DSLR / 激光扫描 | 官方页：[ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/)，文档：[documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)，GitHub：[scannetpp](https://github.com/scannetpp/scannetpp) | 论文引用版本：460 scenes、280K DSLR images、3.7M iPhone RGB-D frames；当前官方文档：1,006 scenes，默认下载约 1.5TB，hi-res DSLR 可到 9TB | Stage 1：3.9%；Stage 2：2.8% | 高保真室内数据，含 laser scan、DSLR、iPhone RGB-D，多模态数据量很大。 |
| ScanNet | 真实室内 RGB-D video | 官方页：[ScanNet](http://www.scan-net.org/ScanNet/)，论文：[CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Dai_ScanNet_Richly-Annotated_3D_CVPR_2017_paper.pdf) | 1,513 个 RGB-D scans，约 2.5M RGB-D views，覆盖 707 个不同空间 | Stage 1：1.9%；Stage 2：2.8% | 含相机位姿、表面重建、语义/实例标注；论文 Appendix A.1 提到 ScanNet 深度为 16-bit PNG 毫米单位，训练前除以 1000 转成米。 |
| Matterport3D | 真实室内 RGB-D / 建筑级 3D scan | 官方页：[Matterport3D](https://niessner.github.io/Matterport/) | 90 个 building-scale scenes，10,800 panoramic views，194,400 RGB-D images | Stage 2：2.6% | 论文 Appendix A.5 将其作为室内 3D scene source，用 Habitat-Sim 渲染跨房间长轨迹。 |
| Gibson | 真实室内空间扫描/虚拟化环境 | 数据库：[Gibson Database](https://gibsonenv.stanford.edu/database/)，下载说明：[GibsonEnv README](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md) | 官方说明：572 models、1,440 floors；标准 split 约 Tiny 8GiB、Medium 21GiB、Full 65GiB、Full+ 89GiB | Stage 2：2.6% | 论文 Appendix A.5 写作约 450 building-scale scans；用于生成长程室内 traversal。 |
| HM3D | 真实室内 Matterport digital twin / 3D scan | 官方页：[AI Habitat HM3D](https://aihabitat.org/datasets/hm3d/)，访问说明：[Matterport HM3D access](https://matterport.com/partners/meta) | 1,000 个高分辨率室内 3D scans / digital twins；约 112.5K m2 navigable space | Stage 2：2.6% | 论文 Appendix A.5 用作室内渲染源；官方访问需 academic/non-commercial 申请。 |

论文 Appendix A.5 进一步说明：作者从 Gibson、Matterport3D、HM3D 三类室内 3D 场景源中生成约 2,800 条序列，每条 1K 到 5K 帧，总计约 14.4TB。这个 14.4TB 是论文自建的渲染派生数据规模，不等同于三个公开数据集的原始下载大小。

## 可筛室内子集的混合场景数据集

这些数据集不是“纯室内”，但公开数据中包含室内环境。若你的目标是只训练室内环境，应按官方环境标签或场景类别过滤，只保留室内部分。

| 数据集 | 类型 | 获取途径 | 规模 / 大小 | 论文中用途 | 室内使用建议 |
|---|---|---|---:|---|---|
| TartanAir | 合成/虚拟 SLAM video，室内+室外混合 | 官方页：[TartanAir](https://tartanair.org/)，Azure Open Datasets：[TartanAir AirSim](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-tartanair-simulation) | Azure 文档记录 V1 数据包含数百条轨迹，约 3TB；含 stereo RGB、depth、segmentation、flow、pose | Stage 1：3.9%；Stage 2：7.6% | 只保留 indoor 或 indoor-like 环境；不要把 outdoor/rural/urban 轨迹混入室内训练统计。 |
| TartanAirV2 | 合成/虚拟 SLAM video，室内+室外混合 | 官方文档：[TartanAir environments](https://tartanair.org/environments.html) | 当前环境表列出 65 个环境，并给出 `Indoor` / `Outdoor` / `Mix` 标签 | Stage 1：5.8%；Stage 2：10.8% | 可按环境表的 `In/Out` 字段筛选 `Indoor`；`Mix` 建议人工复核。 |
| TartanGround | 合成/虚拟地面机器人 video，室内+室外混合 | 官方文档：[TartanGround](https://tartanair.org/tartanground.html) | 63 个场景，878 条轨迹，17.3M RGB images，约 16TB | Stage 1：5.8%；Stage 2：10.8% | 官方类别中包含 Indoor；只筛 Indoor 或明确室内的 infrastructure 场景。 |
| DL3DV / DL3DV-10K | 真实视频，多 POI 场景，室内+室外混合 | 官方页：[DL3DV-10K](https://dl3dv-10k.github.io/DL3DV-10K/)，GitHub：[DL3DV-10K/Dataset](https://github.com/DL3DV-10K/Dataset) | 10,510 个 4K videos，约 51.2M frames，覆盖 65 类 POI，并带 indoor/outdoor 等场景标签 | Stage 1：11.0%；Stage 2：5.7% | 只保留 indoor POI，如餐厅、商场、室内空间等；论文比例是全 DL3DV，不是室内子集比例。 |

## 不纳入本文分析的训练数据

| 数据集 | 论文 stage | 排除原因 |
|---|---|---|
| Mapfree | Stage 1：3.9%；Stage 2：1.5% | 官方数据集是 655 个 outdoor small places of interest。 |
| Waymo、VirtualKITTI、KITTI-360 | Stage 1/Stage 2 依数据集不同 | 自动驾驶/道路环境，不是室内环境。 |
| MatrixCity、MidAir、MegaDepth、MVS-Synth、GTA-SFM、Unreal4K | Stage 1/Stage 2 依数据集不同 | 主要是城市、航拍、户外、in-the-wild 或非室内环境。 |
| WildRGBD、CO3D、Objaverse、Texverse、PointOdyssey、Kubric | Stage 1/Stage 2 依数据集不同 | 主要是对象中心、物体 mesh、点跟踪或非场景级室内环境。 |
| Internal Game | Stage 1：10.6%；Stage 2：10.8% | 内部私有数据，不是公开数据集。 |

## 建议使用优先级

1. 如果只想快速搭建室内训练集：优先 `ScanNet++`、`ScanNet`、`HyperSim`、`SceneNet RGB-D`。
2. 如果要补充第一视角室内长视频：加入 `ADT` 和 `Aria Synthetic Environments`。
3. 如果要复现论文 Stage 2 的长程跨房间能力：使用 `Matterport3D`、`Gibson`、`HM3D` 通过 Habitat-Sim 生成长轨迹，或直接筛选 `TartanAirV2` / `TartanGround` 的室内轨迹。
4. 如果使用 `DL3DV`、`TartanAir` 这类混合数据，务必记录筛选规则；论文给出的 stage ratio 不能直接当作室内子集比例。

## 来源链接

- LingBot-Map 本地论文：[paper-pdf/Geometric Context Transformer for.pdf](paper-pdf/Geometric%20Context%20Transformer%20for.pdf)
- HyperSim：[GitHub](https://github.com/apple/ml-hypersim)，[Apple Research](https://machinelearning.apple.com/research/hypersim)
- SceneNet RGB-D：[official page](https://robotvault.bitbucket.io/scenenet-rgbd.html)
- Aria Synthetic Environments：[dataset page](https://www.projectaria.com/datasets/ase)，[docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset)，[download docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_download_dataset)
- Replica：[GitHub](https://github.com/facebookresearch/replica-dataset)，[arXiv](https://arxiv.org/abs/1906.05797)
- Aria Digital Twin：[dataset page](https://www.projectaria.com/datasets/adt)，[download docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/dataset_download)
- ScanNet：[official data page](http://www.scan-net.org/ScanNet/)，[CVPR 2017 paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Dai_ScanNet_Richly-Annotated_3D_CVPR_2017_paper.pdf)
- ScanNet++：[official site](https://scannetpp.mlsg.cit.tum.de/scannetpp/)，[documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)，[GitHub](https://github.com/scannetpp/scannetpp)
- Matterport3D：[official page](https://niessner.github.io/Matterport/)
- Gibson：[database page](https://gibsonenv.stanford.edu/database/)，[download/size notes](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md)
- HM3D：[AI Habitat page](https://aihabitat.org/datasets/hm3d/)，[Matterport access page](https://matterport.com/partners/meta)
- TartanAir / TartanAirV2：[official docs](https://tartanair.org/)，[environment list](https://tartanair.org/environments.html)，[Azure Open Datasets](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-tartanair-simulation)
- TartanGround：[official docs](https://tartanair.org/tartanground.html)
- DL3DV-10K：[official page](https://dl3dv-10k.github.io/DL3DV-10K/)，[GitHub](https://github.com/DL3DV-10K/Dataset)
