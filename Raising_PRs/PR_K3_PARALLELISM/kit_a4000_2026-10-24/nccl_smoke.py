import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); w = dist.get_world_size(); torch.cuda.set_device(r)
t = torch.ones(1, device="cuda") * (r + 1); dist.all_reduce(t)
big = torch.ones(64 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)
dist.all_reduce(big); torch.cuda.synchronize(); s = time.time()
for _ in range(5): dist.all_reduce(big)
torch.cuda.synchronize(); dt = (time.time() - s) / 5
if r == 0: print(f"world {w} all_reduce sum {t.item()} (expect {w*(w+1)//2}); 128 MiB bf16 all_reduce {dt*1e3:.1f} ms, busbw {2*(w-1)/w*big.numel()*2/dt/1e9:.1f} GB/s")
dist.destroy_process_group()
