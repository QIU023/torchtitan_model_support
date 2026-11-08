# Together：GPU cluster 里的两个 pod（2026-11-07）

依据：Together 文档
- [GPU clusters](https://docs.together.ai/docs/gpu-clusters-quickstart)
- [Kubernetes workloads](https://docs.together.ai/docs/kubernetes-workloads)
- [cluster storage](https://docs.together.ai/docs/cluster-storage)
- [preemptible compute](https://docs.together.ai/docs/preemptible-compute)
- [health checks](https://docs.together.ai/docs/health-checks)

**还没在 Together 上跑过。** 完整流程见 `../PLAYBOOK.md`，这里只放建集群的细节和依据。

## 0. 和 Nebius 不一样的地方

- **IB 进 pod 的方式文档没写：** 文档只说每种卡的节点之间都是 3.2 Tb/s InfiniBand（每个节点 8 条 rail），但没写 pod 怎么拿到 RDMA 设备，也没给 NCCL 设置。清单照 Nebius 的做法开 privileged，流水线第一个任务 `comm` 会告诉你 NCCL 到底走没走 IB。
- **共享存储：** Together 会建一个和共享卷同名的静态 PV（ReadWriteMany，默认 StorageClass），PVC 用 `volumeName` 绑上去即可。
- **抢占式节点：**
  1. 被抢占时先 cordon 节点，发 `TogetherPreemptionNotified` 事件，并给 pod 发 SIGTERM；
  2. 5 分钟后删掉节点；
  3. 有容量时自动补新节点。
  - 预览期没有最短存活时间的保证。
- **权限：** 只有 project admin 能下载凭证；editor 不能 `kubectl exec`，而 `k8s.sh` 要用 exec，所以必须用 admin 的 kubeconfig。

## 1. 建集群（Together 控制台）

- GPU 选 H200-SXM（`HW=h200`），抢占式 16 卡（2 个节点，kit 只支持 2 个节点）；编排选 Kubernetes。
  - 价格先核对卡型：控制台标的 $1.995/GPU·时 和文档里 H100 抢占式的价（$1.99）一致，文档里 H200 抢占式是 $2.99。选的是 H100，就把清单里的 `HW` 改成 `h100`，否则会按 141 GB 的模型和长度跑。
- 共享存储 500 GiB 左右就够（venv、编译 cache、trace），跟集群一起建，并记下卷名。卷能扩不能缩（缩要找客服），文档没写最小值，所以先建小的。单独建的卷可能因为不在同一个数据中心而建集群失败。
- **余额门槛：** 创建时余额要够"抢占式目标的 4 小时 + 按需部分的最短时长 + 同期的存储"，不够就报 `Insufficient credits for cluster creation`。16 张抢占式按 $1.995 算，4 小时是 $127.68，再加存储和按需节点。这只是门槛，实际按用量扣（抢占式每 1 到 2 分钟计一次）。
- **余额用完：** 集群立刻暂停，宽限期内不充值就删除；共享卷立刻停止访问，72 小时内不充值就删数据。扣完之前没有提醒，所以要留余量。集群不能暂停，不用时只能删除；删集群不会删卷。
- 驱动用默认的（截图里是 610 / CUDA 13.3），`setup.sh` 会据此装 cu130 的 torch。
- 文档写着 "Every cluster requires at least one standard node"。如果控制台强制要一个按需节点，最省的是 1 个按需节点加 1 个抢占式节点，两个 pod 一边一个，这时不要打开清单里注释掉的 `nodeSelector`。按需节点按按需价计费（大约是抢占式的两倍），也算进余额门槛；它不会被抢占，两个节点里只有一个会被抢。

## 2. 本机拿到 kubectl

```sh
uv tool install "together[cli]" --upgrade
tg clusters list
tg clusters get-credentials <CLUSTER_ID> --set-default-context     # JSON 模式下不写文件，加 --no-json
kubectl get nodes -L together.ai/compute-class
kubectl get pv                                                      # 找到共享卷同名的那个 PV
```

## 3. 拉起两个 pod

```sh
# 先改 together/p14_pods.yaml：volumeName 填上一步的 PV 名，HW、MN_ROWS（第一次建议 k3_5d）
kubectl create secret generic hf-token --from-literal=token=<hf_token>   # 可选，下载 tokenizer 用
kubectl apply -f phase14_multinode_5D_experiments/kit/together/p14_pods.yaml
bash phase14_multinode_5D_experiments/kit/k8s.sh status
```

之后按 `../PLAYBOOK.md` 第 4 节以后操作。

## 4. `InfiniBand: NO` 时

1. **先确认 IB 本身没问题：** 在控制台跑 Together 的健康检查（Multi-Node NCCL、InfiniBand Write Bandwidth）。阈值是多节点 all_reduce 不低于 330 GB/s、每个设备 ib_write_bw 不低于 320 Gb/s。健康检查要占满 GPU，先 `bash $K8S kill`。
2. **换成节点的网络命名空间：** 打开清单里注释掉的 `hostNetwork: true` 和 `dnsPolicy: ClusterFirstWithHostNet`，把 `MN_IFNAME` 设成 pod 里 `ip -4 addr` 看到的节点网卡名。然后 `kubectl delete job p14`，重新 `apply`，再 `bash $K8S retask`。
3. **还不行就问 Together 支持：** pod 要怎么拿到 RDMA，比如需要申请哪种 device plugin 资源。
4. **NCCL 用到了 RoCE 网口：** `comm` 的结论里列出了 NCCL 用的 HCA。如果其中有 RoCE 的前端网口，在清单里加 `MN_IB_HCA`，只列 IB 的那几个。

## 5. 收尾

1. 先 `bash $K8S collect`；
2. `kubectl delete -f phase14_multinode_5D_experiments/kit/together/p14_pods.yaml`；共享卷默认在集群删除后也保留。
3. 最后在控制台删集群，不勾 "Delete attached shared volume" 就会保留共享卷。保留的卷能挂到同一区域的新集群上接着跑（`../PLAYBOOK.md` 第 8.1 节）；不用时要删，否则会一直计费。
