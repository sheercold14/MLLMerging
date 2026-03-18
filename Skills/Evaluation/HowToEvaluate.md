  1. 环境准备

  参考文档在 eval_vlmevalkit.md。你们这边之前可用的一套环境是：

  conda activate /data/lishichao/env/eval-kit
  cd /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit
  pip install -e .

  pip install 'transformers>=4.45,<5.0' --no-deps
  pip install 'huggingface-hub>=0.34,<1.0'
  pip install accelerate torchvision qwen-vl-utils pyarrow
  pip install nvidia-cusparse-cu12==12.3.1.170 --no-deps

  你们这里的已知坑：

  - transformers 5.x 不兼容
  - huggingface-hub 1.x 不兼容
  - nvidia-cusparse-cu12 很容易被 pip 升回不兼容版本

  2. 官方 TSV 怎么下载

  这几个数据集的官方 TSV URL 已经写死在代码里，见 image_vqa.py：

  - OCRVQA_TEST: https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TEST.tsv
  - OCRVQA_TESTCORE: https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv
  - TextVQA_VAL: https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv
  - ChartQA_TEST: https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv
  - GQA_TestDev_Balanced:
    https://opencompass.openxlab.space/utils/VLMEval/GQA_TestDev_Balanced.tsv
  - VizWiz: https://opencompass.openxlab.space/utils/VLMEval/VizWiz.tsv
  - MathVista_MINI: https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv

  MathVision_MINI 也在同一个文件里定义，你直接跑时会自动按同样逻辑取。

  默认下载位置由 LMUDataRoot() 决定，见 file.py:68：

  - 如果设置了 LMUData 环境变量，就下载到那个目录
  - 否则下载到 VLMEvalKit/LMUData/

  下载逻辑在 image_base.py:78：

  - 如果 ${LMUData}/xxx.tsv 不存在，就自动下载
  - 存在就直接复用

  所以最干净的方式是新建一个空目录专门放官方 TSV，避免混进你之前的自制 TSV：

  mkdir -p /data/lishichao/data/Optmerge/Eval/LMUData_official
  export LMUData=/data/lishichao/data/Optmerge/Eval/LMUData_official

  然后第一次跑评测时，VLMEvalKit 会自动把缺失的官方 TSV 下载进去。

  如果你想手动先下好，也可以直接：

  cd /data/lishichao/data/Optmerge/Eval/LMUData_official

  wget https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv
  wget https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv
  wget https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv
  wget https://opencompass.openxlab.space/utils/VLMEval/GQA_TestDev_Balanced.tsv
  wget https://opencompass.openxlab.space/utils/VLMEval/VizWiz.tsv
  wget https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv
  wget https://opencompass.openxlab.space/utils/VLMEval/MathVision_MINI.tsv

  3. 注册你自己的模型

  如果模型名还没进 supported_VLM，要在 config.py 里加一条。你们之前文档里的写法是：

  'Qwen2-VL-7B-OCR': partial(
      Qwen2VLChat,
      model_path='/data/lishichao/data/model/Qwen2-VL-7B-OCR',
      min_pixels=256*28*28,
      max_pixels=1280*28*28,
  ),

  如果你测别的本地模型，就把 model_path 改成你的路径，名字改成你要传给 --model 的名字。

  4. 执行评测

  入口是 run.py。最直接的命令就是：

  conda activate /data/lishichao/env/eval-kit
  export LMUData=/data/lishichao/data/Optmerge/Eval/LMUData_official
  export CUDA_VISIBLE_DEVICES=0,1,2
  cd /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc-per-node=3 run.py \
    --data MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE VizWiz
  GQA_TestDev_Balanced ChartQA_TEST \
    --model Qwen2-VL-7B-OCR \
    --verbose

  输出会在：

  - VLMEvalKit/outputs

  目录结构一般是：

  - outputs/<模型名>/TYYYYMMDD_G<commit>/

  每个数据集会有：

  - 预测结果：<model>_<dataset>.xlsx
  - 打分结果：<model>_<dataset>_acc.csv 或 <model>_<dataset>_score.csv

  你也可以跑完后汇总：

  python results.py outputs/Qwen2-VL-7B-OCR

  对应脚本在 results.py。

  5. 这几个数据集各自的注意事项

  - TextVQA_VAL / OCRVQA_TESTCORE / ChartQA_TEST / GQA_TestDev_Balanced / VizWiz
    这几个只要 TSV 能下载到，通常可以直接评。
  - MathVista_MINI / MathVision_MINI
    推理能直接跑，但最终评分依赖额外 judge。MathVista 的评估代码里明确要求可用的 OpenAI judge，
    见 image_vqa.py:210。
    代码会检查 OPENAI_API_KEY，见 vlm.py:139。

  如果你要跑这两个，先设：

  export OPENAI_API_KEY=你的key

  必要时还可以显式指定 judge：

  --judge gpt-4o-mini

  6. 最关键的实际建议

  你现在本地已经有一批“自制 TSV”放在 /data/lishichao/data/Optmerge/Eval/LMUData。
  如果你继续把 LMUData 指到那个目录，VLMEvalKit 会直接复用已有同名文件，不会自动替换成官方版。

  所以你自己重测时，建议直接用一个全新的空目录，比如：

  mkdir -p /data/lishichao/data/Optmerge/Eval/LMUData_official
  export LMUData=/data/lishichao/data/Optmerge/Eval/LMUData_official

• 可以，但先说结论：VLMEvalKit 没有一个特别漂亮的“总进度仪表盘”。你看整体进度，主要靠这 3 类信
  号：

  - 终端日志
  - outputs/<model>/T日期_Gcommit/ 里不断生成的文件
  - 每个数据集对应的 .xlsx / _acc.csv / _score.csv / .pkl

  怎么看整体 evaluation 进度

  最直接看当前这次任务的目录，比如：

  outputs

  假设模型名是 Qwen2-VL-7B-Instruct，进去看最新一轮目录：

  cd /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/outputs/Qwen2-VL-7B-Instruct
  ls -dt T* | head

  然后看最新那个目录里落了哪些文件：

  ls -lah /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/outputs/Qwen2-VL-7B-Instruct/
  T2026xxxx_Gxxxxxxx

  你可以这样理解进度：

  - 出现 <model>_<dataset>.xlsx
    说明这个数据集的推理已经跑完并汇总了预测
  - 再出现 <model>_<dataset>_acc.csv
    说明这个数据集的常规评测已经完成
  - 再出现 <model>_<dataset>_score.csv
    说明像 MathVista 这类需要额外 judge 的评分也完成了
  - 出现 *.pkl
    说明还在跑中间过程，或者可以用于断点续跑

  一个实用的看法是数文件：

  find /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/outputs/Qwen2-VL-7B-Instruct/
  T2026xxxx_Gxxxxxxx -maxdepth 1 -type f | sort

  如果你跑的是这 7 个数据集：

  - MathVista_MINI
  - MathVision_MINI
  - TextVQA_VAL
  - OCRVQA_TESTCORE
  - VizWiz
  - GQA_TestDev_Balanced
  - ChartQA_TEST

  那么整体进度大致可以按“每个数据集是否已经有最终评分文件”来判断。

  终端里还有两层进度：

  - run.py 会按 “模型 x 数据集” 顺序逐个跑
  - 具体单个数据集内部，inference.py 里会用 tqdm 跑样本级进度条，API judge 的场景会用
    track_progress_rich

  框架代码怎么走

  主入口是 run.py。

  你可以把它理解成 5 层：

  1. 解析参数
     在 run.py 里，parse_args() 接收：

  - --data
  - --model
  - --work-dir
  - --reuse
  - --judge
  - --mode

  2. 逐个组合调度
     main() 里是双重循环：

  - 外层遍历模型
  - 内层遍历数据集

  每次会新建一个输出目录：

  - outputs/<model>/T{date}_G{commit}/

  这就是你每次跑评测看到的那层目录。

  3. 构建数据集对象
     build_dataset(dataset_name) 会根据名字实例化对应数据集类。
     这些类主要在 image_vqa.py 和同目录其他文件里。

  关键点：

  - DATASET_URL 决定官方下载地址
  - DATASET_MD5 决定官方文件校验
  - prepare_tsv() 决定从哪里找/下 TSV
  - build_prompt() 决定每条样本怎么拼成模型输入
  - evaluate() 决定预测结果怎么打分

  4. 推理
     普通图像任务走 inference.py 里的 infer_data_job()。

  它的流程是：

  - 如果已经有 <model>_<dataset>.xlsx，先把旧预测提出来放到 *_PREV.pkl
  - 按 rank 切分数据
  - 对每个样本调用 model.generate(...)
  - 每 10 条落一次临时 pkl
  - rank 0 最后把所有 rank 的 pkl 合并成一个 .xlsx

  所以你看到：

  - *_PREV.pkl
  - 0...pkl, 1...pkl, 2...pkl
  - 最后变成一个 *.xlsx

  这就是推理阶段的生命周期。

  5. 评测
     推理完成后，run.py 会调用 dataset.evaluate(result_file, **judge_kwargs)。

  这一步会因数据集类型不同而不同：

  - TextVQA / OCRVQA / GQA / ChartQA / VizWiz
    直接本地算分，最后落 _acc.csv
  - MathVista / MathVision
    需要额外 judge model，默认会选 gpt-4o-mini 这一类，在 run.py 里有自动分配逻辑
    这类会额外生成 judge 相关中间文件和最终 _score.csv

  为什么断了能续跑

  核心就在这两层缓存：

  - 推理缓存：*_PREV.pkl 和分 rank 的 *.pkl
  - 评测缓存：像 MathVista 的 judge 临时 pkl

  run.py 里如果加 --reuse，会优先把上一次目录里的结果文件或中间 pkl 复制到当前新目录，再继续跑。
  相关逻辑就在 run.py 里 prev_pred_roots 那段。

  所以：

  - 不加 --reuse，更像“重新开一轮”
  - 加 --reuse，更像“从上次接着补”

  你现在最实用的查看方法

  如果你只是想知道“现在跑到哪了”，我建议看这两处：

  1. 当前输出目录文件数

  find /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/outputs/<模型名>/<最新目录>
  -maxdepth 1 -type f | sort

  2. 实时看新增文件

  watch -n 5 'ls -lah /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/outputs/<模型名>/<
  最新目录>'

  你会很直观地看到：

  - 哪个数据集已经生成 .xlsx
  - 哪个已经生成 _acc.csv
  - 哪个还停在 pkl

