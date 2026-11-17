# Nebius：同一个 GPU cluster 里的 2、4 或 8 个 pod（2026-11-07 写，11-17 加 B300）

依据：Nebius 文档
- [GPU 和 InfiniBand（mk8s）](https://docs.nebius.com/kubernetes/gpu/clusters)
- [GPU cluster 和 InfiniBand fabric](https://docs.nebius.com/compute/clusters/gpu)
- [VM 平台和 preset](https://docs.nebius.com/compute/virtual-machines/types)
- [NCCL test](https://docs.nebius.com/kubernetes/gpu/nccl-test)
- [pod 挂共享文件系统](https://docs.nebius.com/kubernetes/storage/filesystem-over-csi)
- [抢占式 VM](https://docs.nebius.com/compute/virtual-machines/preemptible)
- [node group](https://docs.nebius.com/kubernetes/node-groups/manage)
- [NVIDIA AICR：Nebius B300 的 recipe](https://docs.nvidia.com/aicr/integrator-guide/nebius-b-300-inference)

**还没在 Nebius 上跑过，B300 也还没跑过。** pod 起来以后的流程见 `../K8S_RUN.md`，快速路径和各个开关见 `../PLAYBOOK.md` 第 0 节。

## 0. 组网是怎么通的

- **同一张 IB 网：** node group 用 `--template-gpu-cluster-id` 挂到同一个 GPU cluster，所有节点就在同一张 InfiniBand 网上。
- **pod 之间直接互通：** pod 走 k8s 的 pod 网络（eth0）互通，不需要 overlay 和 SSH：
  - rendezvous 和 NCCL bootstrap 走 eth0，`MASTER_ADDR=p14-0.p14`（headless Service 提供）；
  - NCCL 数据走 IB：容器是 privileged，能看到 IB 设备；容器里的 libibverbs 由 `setup_node.sh` 安装。
- **NCCL 设置：** 照 Nebius 官方 NCCL 测试，pod 里设 `MN_NET=nebius`，`kit/env.sh` 会设好：
  - `NCCL_SOCKET_IFNAME=eth0`、`UCX_NET_DEVICES=eth0`、`SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1`、`NCCL_COLLNET_ENABLE=0`；
  - `NCCL_IB_HCA`：只列 link layer 是 InfiniBand 的口（`=mlx5_0,mlx5_1,...`）。B300 节点还有一块 BlueField-3 前端网卡，也叫 mlx5，只写前缀 `mlx5` 会把它也交给 NCCL；一个 IB 口都找不到时才退回 `mlx5`。
- **共享存储：** Nebius 的 shared filesystem 挂到 node group 上，再由 CSI chart `csi-mounted-fs-path` 给 pod 一个 ReadWriteMany 的 PVC。venv、kit、结果和流水线状态都放在上面，所以 pod 被抢占后不会丢。

## 1. 建资源（本机，Nebius CLI）

- **平台**（GPU cluster 只支持这些 preset）：
  - gpu-b300-sxm：uk-south1、eu-west2、us-north1，preset `8gpu-192vcpu-2768gb`，fabric 依次是 `uk-south1-a`、`eu-west2-a`、`us-north1-a`。有一页文档把 uk-south1 标成私有区域，建之前在控制台或 `nebius compute platform list` 核对；
  - gpu-h200-sxm：eu-north1、eu-north2、eu-west1、us-central1，preset `8gpu-128vcpu-1600gb`；
  - gpu-b200-sxm：us-central1，preset `8gpu-160vcpu-1792gb`；
  - gpu-b200-sxm-a：me-west1，preset 同 B200；
  - gpu-h100-sxm：eu-north1。
  - 先用 B300（`HW=b300`，清单默认）。改用 H200 时，清单里 `HW=h200`，资源改成 cpu 96、memory 1200Gi。
- **前提：** mk8s 集群本身按 Nebius 文档建好；GPU cluster、filesystem、node group 和 mk8s 集群必须在同一个 project、同一个区域。node group 建好以后平台、preset 和 GPU cluster 都不能改，要换只能新建。

```sh
# GPU cluster（fabric 按区域选，B300 见上）
export GPU_CLUSTER_ID=$(nebius compute gpu-cluster create --name p14-gpus \
  --infiniband-fabric us-north1-a --format jsonpath='{.metadata.id}')
# 共享文件系统（按 GiB 计费，文档写的是 $0.08/GiB/月；约 500 GiB 够用）
nebius compute filesystem create --name p14-fs --size-gibibytes 500 --type network_ssd --block-size-bytes 4096
export FS_ID=<上一步输出的 id>
# node group：GPU cluster + 共享文件系统（cloud-init 挂到 /mnt/data；nofail 不能省）
export MOUNT_POINT=/mnt/data MOUNT_TAG=p14fs
export USER_DATA=$(jq -Rrs '.' <<EOF
runcmd:
  - sudo mkdir -p $MOUNT_POINT
  - sudo mount -t virtiofs $MOUNT_TAG $MOUNT_POINT
  - printf "%s %s virtiofs defaults,nofail 0 2\n" "$MOUNT_TAG" "$MOUNT_POINT" | sudo tee -a /etc/fstab
EOF
)
nebius mk8s node-group create --parent-id <mk8s 集群 id> --name p14-ib --fixed-node-count 2 \
  --template-resources-platform gpu-b300-sxm --template-resources-preset 8gpu-192vcpu-2768gb \
  --template-gpu-cluster-id $GPU_CLUSTER_ID --template-gpu-settings-drivers-preset cuda13.0 \
  --template-filesystems "[{\"existing_filesystem\": {\"id\": \"$FS_ID\"}, \"attach_mode\": \"READ_WRITE\", \"mount_tag\": \"$MOUNT_TAG\"}]" \
  --template-cloud-init-user-data "$USER_DATA" \
  --template-preemptible --template-follows-spot-price
```

- **节点数：** `--fixed-node-count` 是 2、4 或 8，和后面 `k8s.sh apply` 的数一致。
- **抢占式那一行：** `--template-preemptible --template-follows-spot-price` 就是开抢占式。文档没说抢占式能不能挂 GPU cluster。建之前用 `nebius compute platform list` 看对应平台有没有 `allowed_for_preemptibles: true`，建不出来就去掉这一行，改用按需。
- **启动盘：** 不用太大，venv、cache 和结果都在共享文件系统上。
- **CSI chart**（kubeconfig 指向这个集群之后，在本机执行）：

```sh
helm pull oci://cr.eu-north1.nebius.cloud/mk8s/helm/csi-mounted-fs-path --version 0.1.10
helm upgrade csi-mounted-fs-path ./csi-mounted-fs-path-0.1.10.tgz --install \
  --set dataDir=/mnt/data/csi-mounted-fs-path-data/
kubectl get storageclass csi-mounted-fs-path-sc
```

## 2. B300 和 kit 已经跑过的卡有什么不同

| 项 | B300 节点 | kit 里对应的处理 |
|---|---|---|
| 显存 | 每卡 270 GB HBM3e | `HW=b300` 选 270g 缩小版：`k3_16x270g` / `k3_32x270g` / `k3_64x270g`，层数和专家数与 80g、141g 同名版本相同，只加宽（`PLAYBOOK.md` 第 0.1 节）。M3 没有 270g 版本，2 节点时用 `m3_16x141g` |
| 序列长度 | | 和 H200 相同的长度表；快速路径固定 16K |
| 网络 | 每卡一个 800 Gb/s InfiniBand 口（ConnectX-8），另有一块 BlueField-3 前端网卡 | `NCCL_IB_HCA` 只列 IB 口（第 0 节）；preflight 的 RDMA 阈值是每个口链路速率的 80%，all_reduce 阈值按链路速率从 400G 的 300 GB/s 放大到 600 GB/s |
| 驱动和 torch | drivers preset `cuda13.0`；AICR 的 Nebius B300 recipe 固定驱动 595.91.07 | `setup.sh` 按驱动的 CUDA 主版本装 torch nightly 的 cu130 版本 |
| 算力口径 | | titan 的 `get_peak_flops` 把 B300 记成 2.25 PFLOPS bf16 稠密，和 B200 相同 |
| CPU、内存 | preset 192 vCPU、2768 GB | 清单的 requests / limits 是 cpu 160、memory 2400Gi，给节点上的系统组件留余量 |

还没核实的三件事，第一轮就能看出来：
- **sm_103 上的 kernel：** K3 的 flex attention、KDA 的 Triton 和 CuTeDSL kernel、`ext_build.sh` 编的 DeepEP / HybridEP 都没在 B300 上跑过。所以第一轮先只跑一行（第 3 节）。
- **800G 的阈值：** 600 GB/s 是按比例推的，没实测，所以清单里 `MN_PREFLIGHT=warn`：只报告，不拦矩阵。第一轮的 `preflight/VERDICT.txt` 就是这批节点的实测，之后可以改回 `1`。
- **抢占式能不能挂 GPU cluster：** 见第 1 节。

## 3. 拉起 pod

先确认 `nebius/p14_pods.yaml` 里的三处：
- `HW`：`b300`；
- 资源：要和 node group 的 preset 匹配；
- `MN_COMPILE`：`0` 是 eager；`1` 是每一行都编译，冷编译每种布局多十几分钟，这时矩阵里只开 compile 的那一行会被跳过。

```sh
K8S=phase14_multinode_5D_experiments/kit/k8s.sh
M=phase14_multinode_5D_experiments/kit/nebius/p14_pods.yaml
kubectl create secret generic hf-token --from-literal=token=<hf_token>   # 可选，下载 tokenizer 用
# 第一轮 B300：只跑 k3_A，确认 sm_103 上能起来、能走步
bash $K8S apply 2 k3_A $M
bash $K8S status
# k3_A 的 measure 跑通后，放开整个矩阵（或 +core）
bash $K8S scope MN_ROWS=; bash $K8S retask
```

之后按 `../K8S_RUN.md` 操作。4、8 个节点把 `apply` 的 2 换成 4、8（node group 的 `--fixed-node-count` 要一致）。

## 4. `InfiniBand: NO` 时

依次检查五件事：
- pod 是不是 privileged；
- `bash $K8S shell 0` 进去后，`ibv_devinfo -l` 能不能列出 mlx5 设备；
- `cat /sys/class/infiniband/*/ports/1/link_layer` 里有没有 `InfiniBand`，`node0.log` 里 `NCCL_IB_HCA` 是不是只列了这些口；
- `setup_node_*.log` 里 rdma-core 有没有装上；
- node group 有没有挂上 GPU cluster。

## 5. 收尾

1. 先 `bash $K8S collect`；
2. `kubectl delete -f phase14_multinode_5D_experiments/kit/nebius/p14_pods.yaml`，这会删掉 Job、Service 和 PVC；文件系统里的数据在 CSI 的 dataDir 下，删除 PVC 后会不会被清掉，文档没写，所以先 collect；
3. 最后删 node group 和文件系统（`nebius compute filesystem delete --id $FS_ID`），否则会一直计费。

## 6. Nebius 的抢占行为

- 抢占式 VM 被抢占时只停机，不删除，也不会自动重启。停机前 60 秒发 SIGTERM；挂着的磁盘和共享文件系统的数据都保留。
- 文档没写 mk8s 的抢占式节点被抢占后会怎样。按文档的健康检查规则推测（没核实）：节点 NotReady 超过 15 分钟会被删掉，再建一个新节点；新节点还是抢占式，要等容量和价格。
- 节点一直不回来时，可以在控制台或用 `nebius compute instance start --id <VM_ID>` 手动启动那台 VM。
