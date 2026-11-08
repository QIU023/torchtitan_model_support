# Nebius：同一个 GPU cluster 里的两个 pod（2026-11-07）

依据：Nebius 文档
- [GPU 和 InfiniBand](https://docs.nebius.com/kubernetes/gpu/clusters)
- [NCCL test](https://docs.nebius.com/kubernetes/gpu/nccl-test)
- [pod 挂共享文件系统](https://docs.nebius.com/kubernetes/storage/filesystem-over-csi)
- [抢占式 VM](https://docs.nebius.com/compute/virtual-machines/preemptible)
- [node group](https://docs.nebius.com/kubernetes/node-groups/manage)

**还没在 Nebius 上跑过。** pod 起来以后的流程见 `../K8S_RUN.md`。

## 0. 组网是怎么通的

- **同一张 IB 网：** node group 用 `--template-gpu-cluster-id` 挂到同一个 GPU cluster，两个节点就在同一张 InfiniBand 网上。
- **pod 之间直接互通：** 两个 pod 走 k8s 的 pod 网络（eth0）互通，不需要 overlay 和 SSH：
  - rendezvous 和 NCCL bootstrap 走 eth0，`MASTER_ADDR=p14-0.p14`（headless Service 提供）；
  - NCCL 数据走 IB（`NCCL_IB_HCA=mlx5`）：容器是 privileged，能看到 IB 设备；容器里的 libibverbs 由 `setup_node.sh` 安装。
- **NCCL 设置：** 照 Nebius 官方 NCCL 测试，pod 里设 `MN_NET=nebius`，`kit/env.sh` 会设好 `NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_HCA=mlx5`、`UCX_NET_DEVICES=eth0`、`SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1`、`NCCL_COLLNET_ENABLE=0`。
- **共享存储：** Nebius 的 shared filesystem 挂到 node group 上，再由 CSI chart `csi-mounted-fs-path` 给 pod 一个 ReadWriteMany 的 PVC。venv、kit、结果和流水线状态都放在上面，所以 pod 被抢占后不会丢。

## 1. 建资源（本机，Nebius CLI）

- **平台**（GPU cluster 只支持这些 preset）：
  - gpu-h100-sxm：eu-north1；
  - gpu-h200-sxm：eu-north1、eu-north2、eu-west1、us-central1，preset `8gpu-128vcpu-1600gb`；
  - gpu-b200-sxm：us-central1，preset `8gpu-160vcpu-1792gb`；
  - gpu-b200-sxm-a：me-west1，preset 同 B200。
  - 推荐 H200（`HW=h200`）。
- **前提：** mk8s 集群本身按 Nebius 文档建好；GPU cluster、filesystem、node group 和 mk8s 集群必须在同一个 project、同一个区域。

```sh
# GPU cluster（fabric 按区域和卡型选）
export GPU_CLUSTER_ID=$(nebius compute gpu-cluster create --name p14-gpus \
  --infiniband-fabric <fabric> --format jsonpath='{.metadata.id}')
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
  --template-resources-platform gpu-h200-sxm --template-resources-preset 8gpu-128vcpu-1600gb \
  --template-gpu-cluster-id $GPU_CLUSTER_ID --template-gpu-settings-drivers-preset cuda13.0 \
  --template-filesystems "[{\"existing_filesystem\": {\"id\": \"$FS_ID\"}, \"attach_mode\": \"READ_WRITE\", \"mount_tag\": \"$MOUNT_TAG\"}]" \
  --template-cloud-init-user-data "$USER_DATA" \
  --template-preemptible --template-follows-spot-price
```

- **抢占式那一行：** `--template-preemptible --template-follows-spot-price` 就是开抢占式。文档没说抢占式能不能挂 GPU cluster。建之前用 `nebius compute platform list` 看对应平台有没有 `allowed_for_preemptibles: true`，建不出来就去掉这一行，改用按需。
- **启动盘：** 不用太大，venv、cache 和结果都在共享文件系统上。
- **CSI chart**（kubeconfig 指向这个集群之后，在本机执行）：

```sh
helm pull oci://cr.eu-north1.nebius.cloud/mk8s/helm/csi-mounted-fs-path --version 0.1.10
helm upgrade csi-mounted-fs-path ./csi-mounted-fs-path-0.1.10.tgz --install \
  --set dataDir=/mnt/data/csi-mounted-fs-path-data/
kubectl get storageclass csi-mounted-fs-path-sc
```

## 2. 拉起两个 pod

```sh
# 先改 nebius/p14_pods.yaml 里的 HW（h100 / h200 / b200）和 MN_ROWS（第一次建议 k3_5d）
kubectl create secret generic hf-token --from-literal=token=<hf_token>   # 可选，下载 tokenizer 用
kubectl apply -f phase14_multinode_5D_experiments/kit/nebius/p14_pods.yaml
bash phase14_multinode_5D_experiments/kit/k8s.sh status
```

之后按 `../K8S_RUN.md` 操作。

## 3. `InfiniBand: NO` 时

依次检查四件事：
- pod 是不是 privileged；
- `bash $K8S shell 0` 进去后，`ibv_devinfo -l` 能不能列出 mlx5 设备；
- `setup_node_*.log` 里 rdma-core 有没有装上；
- node group 有没有挂上 GPU cluster。

## 4. 收尾

1. 先 `bash $K8S collect`；
2. `kubectl delete -f phase14_multinode_5D_experiments/kit/nebius/p14_pods.yaml`，这会删掉 Job、Service 和 PVC；文件系统里的数据在 CSI 的 dataDir 下，删除 PVC 后会不会被清掉，文档没写，所以先 collect；
3. 最后删 node group 和文件系统（`nebius compute filesystem delete --id $FS_ID`），否则会一直计费。

## 5. Nebius 的抢占行为

- 抢占式 VM 被抢占时只停机，不删除，也不会自动重启。停机前 60 秒发 SIGTERM；挂着的磁盘和共享文件系统的数据都保留。
- 文档没写 mk8s 的抢占式节点被抢占后会怎样。按文档的健康检查规则推测（没核实）：节点 NotReady 超过 15 分钟会被删掉，再建一个新节点；新节点还是抢占式，要等容量和价格。
- 节点一直不回来时，可以在控制台或用 `nebius compute instance start --id <VM_ID>` 手动启动那台 VM。
