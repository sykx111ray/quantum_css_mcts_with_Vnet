# 复现实验操作指南（傻瓜版）：看有无 ValueNet 时，电路是更快还是更准

目标：复现并比较“无 ValueNet 的 Baseline MCTS”与“有 ValueNet 的 ValueNet-assisted MCTS”，看它们在以下维度上谁更好：

- 速度：运行时间（runtime_s）
- 效率：电路门数（CNOT）和深度（depth）
- 准确率/正确性：是否有效（is_valid）、逻辑是否正确（logical_ok）、syndrome 错误（x_syndrome / z_syndrome）

这个实验已经被封装在脚本 [experiment_22_value_vs_baseline.py](experiment_22_value_vs_baseline.py) 中。它会自动同时跑两组：

- Baseline：不使用 ValueNet
- ValueNet：使用默认 checkpoint 进行辅助搜索

因此，你只需要执行一次主命令即可。

---

## 1. 先进入项目目录

在 PowerShell 里执行：

```powershell
cd /d D:\Vnet
```

如果你还没激活虚拟环境，执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

---

## 2. 先确认脚本参数

执行下面这条命令，确认脚本入口和参数都正常：

```powershell
python .\experiment_22_value_vs_baseline.py --help
```

如果正常，你会看到参数列表，例如：

- --runs
- --iterations
- --rollouts
- --output_dir
- --device
- --trace

---

## 3. 先跑一个“快速试跑”

这是最稳妥的起步方式，避免一次跑太久。

```powershell
python .\experiment_22_value_vs_baseline.py --runs 1 --iterations 200 --rollouts 5 --output_dir .\results\exp22_quick --device cpu
```

说明：

- --runs 1：只跑 1 轮，速度快
- --iterations 200：每组搜索迭代次数少一点
- --rollouts 5：每个节点的 rollout 次数少一点
- --output_dir .\results\exp22_quick：结果输出到这个目录
- --device cpu：优先用 CPU，兼容性最好

这一步的作用是：先确认实验能正常跑通。

---

## 4. 如果快速试跑成功，执行正式复现

正式复现建议用下面这条命令：

```powershell
python .\experiment_22_value_vs_baseline.py --runs 20 --iterations 2000 --rollouts 50 --output_dir .\results\exp22_full --device cpu
```

说明：

- --runs 20：每组跑 20 次，统计更稳定
- --iterations 2000：搜索更充分
- --rollouts 50：每个节点做更多 rollout，结果更稳
- --output_dir .\results\exp22_full：结果保存到这个目录

这一步就是你真正想要的“对比实验”。

---

## 5. 如果你想看更细的中间行为，再加 trace

如果你想看每一步选择/评估/回传的行为，可以加上：

```powershell
python .\experiment_22_value_vs_baseline.py --runs 20 --iterations 2000 --rollouts 50 --output_dir .\results\exp22_full --device cpu --trace --trace_output .\results\exp22_full\exp22_trace.csv --trace_summary .\results\exp22_full\exp22_trace_summary.txt
```

这会额外生成：

- trace CSV：记录选择和评估信息
- trace summary：更容易看出行为差异

---

## 6. 结果怎么读

脚本运行完成后，会在你指定的 output_dir 里生成这些文件：

- exp22_raw.csv
- exp22_statistics.csv
- exp22_convergence.csv
- exp22_summary.txt

如果你开了 --trace，还会多出：

- exp22_trace.csv
- exp22_trace_summary.txt

---

## 7. 重点看哪些列

### 速度

看：

- runtime_s

它表示每次实验的总运行时间。

### 效率

看：

- cnot
- depth

它们分别表示：

- cnot：门数，越少越好
- depth：电路深度，越浅越好

### 准确率/正确性

看：

- is_valid
- logical_ok
- x_syndrome
- z_syndrome

其中：

- is_valid：是否通过有效性验证
- logical_ok：逻辑是否正确
- syndrome 错误越小越好

---

## 8. 最直接的结论读取方式

### 方式 A：直接看总结文件

```powershell
Get-Content .\results\exp22_full\exp22_summary.txt
```

这个文件会直接给出 Baseline 和 ValueNet 的平均结果对比。

### 方式 B：看统计表格

```powershell
Import-Csv .\results\exp22_full\exp22_statistics.csv | Format-Table
```

你要关注的是：

- runtime_s 是否更小
- cnot 是否更少
- depth 是否更小
- valid_rate 是否更高
- syndrome 错误是否更低

---

## 9. 一句话结论判断标准

如果你想简单判断“有 ValueNet 是否更好”，可以这样看：

- 更快：runtime_s 更小
- 更高效：cnot 和 depth 更小
- 更准确：is_valid / logical_ok 更高，syndrome error 更低

如果 ValueNet 那组同时满足：

- 更短运行时间
- 更少 gate / 更浅深度
- 更高有效率和更低错误率

那就说明有 ValueNet 的版本更好。

---

## 10. 最推荐的实操顺序

1. 先跑快速试跑：

```powershell
python .\experiment_22_value_vs_baseline.py --runs 1 --iterations 200 --rollouts 5 --output_dir .\results\exp22_quick --device cpu
```

2. 确认能出结果后，再跑正式实验：

```powershell
python .\experiment_22_value_vs_baseline.py --runs 20 --iterations 2000 --rollouts 50 --output_dir .\results\exp22_full --device cpu
```

3. 查看结果：

```powershell
Get-Content .\results\exp22_full\exp22_summary.txt
```

---

如果你想，我也可以下一步直接把这个说明再整理成“适合发给别人复现的 README 版”，或者顺手帮你写一个 one-click 的 .bat 脚本。