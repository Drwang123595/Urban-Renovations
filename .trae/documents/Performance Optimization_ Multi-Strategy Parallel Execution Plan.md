我理解您的需求，核心在于确保**完全隔离**和**真正的并行**。就如同您所比喻的，这相当于让 5 个独立的研究员（策略）同时拿到同一份试卷（论文），各自在自己的房间里独立作答，互不干扰，最后同时交卷。

以下是深化的实施计划：

## 1. 核心机制：策略级完全并行与隔离 (The "5 People" Model)
我们将采用 **“论文级串行，策略级并行”** 的架构。
- **独立身份 (Identity)**: 
  - 在程序启动时，实例化多个独立的策略对象（`SingleStrategy`, `CoTStrategy` 等）。每个对象都有自己独立的内存空间、Prompt 生成器和状态维护，绝无共享变量，从根源上杜绝逻辑混乱。
- **并行分发 (Dispatch)**:
  - 当读取到一篇论文（Title + Abstract）时，程序会**同时**按下所有策略的“启动键”。
  - 使用 `ThreadPoolExecutor` 瞬间创建 N 个独立的线程（N=策略数）。
  - **线程 1** 拿着论文去找 Single 策略跑单轮对话。
  - **线程 2** 拿着论文去找 CoT 策略跑思维链推理。
  - ...
  - **线程 5** 拿着论文去找 Reflection 策略跑反思逻辑。
  - 这 N 个线程并行运行，互不等待，互不通信。
- **安全回收 (Collection)**:
  - 主程序在“大厅”等待，每当一个房间（线程）做完了，就接过它的结果，放入该策略专属的文件夹（List）中。
  - 最终生成 N 个独立的文件，确保数据纯净。

## 2. 代码实现细节
### 修改 `src/data_processor.py`
- **初始化 (`__init__`)**:
  - 接收 `strategies` 列表。
  - 建立 `self.strategies` 字典：`{'single': instance_A, 'cot': instance_B, ...}`。
- **批处理 (`run_batch`)**:
  - **文件准备**: 为每个策略预先定义独立的输出路径 `output/single_xxx.xlsx`, `output/cot_xxx.xlsx` 等。
  - **双层循环结构**:
    - **外层 (串行)**: 遍历每一篇论文。
    - **内层 (并行)**:
      ```python
      # 伪代码示意
      with ThreadPoolExecutor(max_workers=len(self.strategies)) as executor:
          # 同时分发任务
          future_to_name = {
              executor.submit(strategy.process, title, abstract): name 
              for name, strategy in self.strategies.items()
          }
          
          # 独立收集结果
          for future in as_completed(future_to_name):
              name = future_to_name[future]
              result = future.result() # 捕获该策略的独立产出
              self.results[name].append(result) # 存入专属列表
      ```
  - **独立保存**: 定期将 `self.results['single']` 写入文件 A，`self.results['cot']` 写入文件 B。

### 修改 `scripts/main.py`
- **用户交互**: 更新选择逻辑，支持 "1,2,5" 或 "all" 这种多选输入。
- **参数传递**: 将解析后的策略列表（如 `['single', 'stepwise', 'reflection']`）传给 `DataProcessor`。

## 3. 预期效果
- **效率**: 总耗时 ≈ 最慢的那个策略的耗时（而不是所有策略耗时之和）。
- **质量**: 策略之间绝对物理隔离，不会出现 CoT 的中间推理过程“污染”到 Single 的结果中。
- **产出**: 一次运行，同时获得多份不同维度的分析报告，便于直接横向对比。
