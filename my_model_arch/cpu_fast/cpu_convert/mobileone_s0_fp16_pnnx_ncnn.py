# =============================================================================
# NeuSpeech Institute, NeuGaze Project
# Copyright (c) 2024 Yiqian Yang
#
# This code is part of the NeuGaze project developed at NeuSpeech Institute.
# Author: Yiqian Yang
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
# International License. To view a copy of this license, visit:
# http://creativecommons.org/licenses/by-nc/4.0/
# =============================================================================

import numpy as np
import ncnn
import torch
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# 全局变量，避免重复加载模型文件
net = None
global_extractor = None

def init_model():
    """初始化模型，只需要调用一次"""
    global net, global_extractor
    net = ncnn.Net()
    net.load_param("mobileone_s0_fp16_pnnx.ncnn.param")
    net.load_model("mobileone_s0_fp16_pnnx.ncnn.bin")
    global_extractor = net.create_extractor()  # 创建一个全局的extractor用于对比测试
    print("模型初始化完成")

class AsyncInferenceEngine:
    """异步推理引擎 - 模仿MediaPipe的异步处理方式"""
    
    def __init__(self, num_workers=2):
        self.num_workers = num_workers
        self.input_queue = queue.Queue(maxsize=3)  # 限制队列大小避免内存爆炸
        self.output_queue = queue.Queue()
        self.workers = []
        self.running = False
        
        # 为每个worker创建独立的net和extractor
        self.nets = []
        for _ in range(num_workers):
            worker_net = ncnn.Net()
            worker_net.load_param("mobileone_s0_fp16_pnnx.ncnn.param")
            worker_net.load_model("mobileone_s0_fp16_pnnx.ncnn.bin")
            self.nets.append(worker_net)
    
    def start(self):
        """启动异步处理"""
        self.running = True
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """停止异步处理"""
        self.running = False
        # 清空队列
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
    
    def _worker(self, worker_id):
        """工作线程"""
        net = self.nets[worker_id]
        
        while self.running:
            try:
                # 获取输入数据
                frame_id, input_data = self.input_queue.get(timeout=0.1)
                
                # 执行推理
                start_time = time.perf_counter_ns()
                
                # 重用extractor方式（更快但可能有缓存问题）
                ex = net.create_extractor()
                ex.input("in0", ncnn.Mat(input_data.squeeze(0).numpy()))
                
                _, out0 = ex.extract("out0")
                out0_tensor = torch.from_numpy(np.array(out0)).unsqueeze(0)
                
                _, out1 = ex.extract("out1")
                out1_tensor = torch.from_numpy(np.array(out1)).unsqueeze(0)
                
                result = (out0_tensor, out1_tensor)
                
                end_time = time.perf_counter_ns()
                inference_time = (end_time - start_time) / 1_000_000
                
                # 输出结果
                self.output_queue.put((frame_id, result, inference_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    def process_async(self, frame_id, input_data):
        """异步处理输入"""
        try:
            self.input_queue.put_nowait((frame_id, input_data))
            return True
        except queue.Full:
            return False  # 队列满了，丢弃这一帧
    
    def get_result(self, timeout=0.001):
        """获取结果"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

def test_inference_new_extractor(input_data):
    """每次创建新extractor的推理函数"""
    global net
    out = []
    
    # 每次创建新的extractor
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(input_data.squeeze(0).numpy()))
    
    _, out0 = ex.extract("out0")
    out0_tensor = torch.from_numpy(np.array(out0)).unsqueeze(0)
    out.append(out0_tensor)
    
    _, out1 = ex.extract("out1")
    out1_tensor = torch.from_numpy(np.array(out1)).unsqueeze(0)
    out.append(out1_tensor)
    
    result = tuple(out) if len(out) > 1 else out[0]
    
    # 强制计算完成
    if isinstance(result, tuple):
        _ = result[0].sum().item()
        _ = result[1].sum().item()
    else:
        _ = result.sum().item()
    
    return result

def test_inference_reuse_extractor(input_data):
    """重用extractor的推理函数（仅用于性能对比）"""
    global global_extractor
    out = []
    
    # 重用全局extractor
    global_extractor.input("in0", ncnn.Mat(input_data.squeeze(0).numpy()))
    
    _, out0 = global_extractor.extract("out0")
    out0_tensor = torch.from_numpy(np.array(out0)).unsqueeze(0)
    out.append(out0_tensor)
    
    _, out1 = global_extractor.extract("out1")
    out1_tensor = torch.from_numpy(np.array(out1)).unsqueeze(0)
    out.append(out1_tensor)
    
    result = tuple(out) if len(out) > 1 else out[0]
    
    # 强制计算完成
    if isinstance(result, tuple):
        _ = result[0].sum().item()
        _ = result[1].sum().item()
    else:
        _ = result.sum().item()
    
    return result

def test_async_inference(num_frames=100):
    """测试异步推理性能"""
    print(f"=== 异步推理测试 ===")
    
    # 初始化异步引擎
    engine = AsyncInferenceEngine(num_workers=2)
    engine.start()
    
    # 准备输入数据
    torch.manual_seed(0)
    input_data = torch.rand(1, 3, 448, 448, dtype=torch.float)
    
    # 预热
    print("预热中...")
    for i in range(10):
        engine.process_async(i, input_data)
        time.sleep(0.001)
    
    # 清空结果队列
    while engine.get_result() is not None:
        pass
    
    print("开始异步测试...")
    start_time = time.perf_counter()
    
    # 发送所有帧
    sent_frames = 0
    for i in range(num_frames):
        if engine.process_async(i, input_data):
            sent_frames += 1
        time.sleep(0.001)  # 模拟摄像头帧间隔
    
    # 收集结果
    results = []
    received_frames = 0
    timeout_start = time.perf_counter()
    
    while received_frames < sent_frames and (time.perf_counter() - timeout_start) < 5.0:
        result = engine.get_result(timeout=0.1)
        if result is not None:
            frame_id, output, inference_time = result
            results.append(inference_time)
            received_frames += 1
    
    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000
    
    engine.stop()
    
    if results:
        avg_inference_time = np.mean(results)
        throughput = received_frames / (total_time / 1000)
        
        print(f"发送帧数: {sent_frames}")
        print(f"接收帧数: {received_frames}")
        print(f"总时间: {total_time:.1f} ms")
        print(f"平均推理时间: {avg_inference_time:.3f} ms")
        print(f"吞吐量: {throughput:.1f} FPS")
        print(f"理论最大FPS: {1000/avg_inference_time:.1f}")
    
    return results

def test_inference():
    """原始的推理函数，保持兼容性"""
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 448, 448, dtype=torch.float)
    out = []
    print(f'in0.shape:{in0.shape},dtype:{in0.dtype}')
    with ncnn.Net() as net:
        net.load_param("mobileone_s0_fp16_pnnx.ncnn.param")
        net.load_model("mobileone_s0_fp16_pnnx.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))
            _, out1 = ex.extract("out1")
            out.append(torch.from_numpy(np.array(out1)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

def benchmark_comparison(num_runs=100):
    """对比测试：新建extractor vs 重用extractor"""
    print(f"=== 性能对比测试：新建 vs 重用 Extractor ===")
    
    # 初始化模型
    print("初始化模型...")
    init_model()
    
    # 准备输入数据
    torch.manual_seed(0)
    input_data = torch.rand(1, 3, 448, 448, dtype=torch.float)
    print(f'input_data.shape:{input_data.shape},dtype:{input_data.dtype}')
    
    # 预热
    print("预热中...")
    for _ in range(10):
        test_inference_new_extractor(input_data)
        test_inference_reuse_extractor(input_data)
    
    # 测试1：每次创建新extractor
    print(f"\n测试1：每次创建新extractor（官方推荐方式）")
    times_new = []
    for i in range(num_runs):
        start_time = time.perf_counter_ns()
        result = test_inference_new_extractor(input_data)
        end_time = time.perf_counter_ns()
        times_new.append((end_time - start_time) / 1_000_000)  # 转换为毫秒
        
        if (i + 1) % 20 == 0:
            print(f"已完成 {i + 1}/{num_runs} 次")
    
    # 测试2：重用extractor
    print(f"\n测试2：重用extractor（可能有缓存问题）")
    times_reuse = []
    for i in range(num_runs):
        start_time = time.perf_counter_ns()
        result = test_inference_reuse_extractor(input_data)
        end_time = time.perf_counter_ns()
        times_reuse.append((end_time - start_time) / 1_000_000)  # 转换为毫秒
        
        if (i + 1) % 20 == 0:
            print(f"已完成 {i + 1}/{num_runs} 次")
    
    # 计算统计信息
    times_new = np.array(times_new)
    times_reuse = np.array(times_reuse)
    
    avg_new = np.mean(times_new)
    avg_reuse = np.mean(times_reuse)
    
    print(f"\n=== 对比结果 ===")
    print(f"每次创建新extractor:")
    print(f"  平均时间: {avg_new:.3f} ms")
    print(f"  最快时间: {np.min(times_new):.3f} ms")
    print(f"  最慢时间: {np.max(times_new):.3f} ms")
    print(f"  标准差: {np.std(times_new):.3f} ms")
    print(f"  平均FPS: {1000/avg_new:.1f}")
    
    print(f"\n重用extractor:")
    print(f"  平均时间: {avg_reuse:.3f} ms")
    print(f"  最快时间: {np.min(times_reuse):.3f} ms")
    print(f"  最慢时间: {np.max(times_reuse):.3f} ms")
    print(f"  标准差: {np.std(times_reuse):.3f} ms")
    print(f"  平均FPS: {1000/avg_reuse:.1f}")
    
    speedup = avg_new / avg_reuse
    print(f"\n性能差异:")
    if speedup > 1:
        print(f"  重用extractor比新建快 {speedup:.2f}x ({((speedup-1)*100):.1f}%)")
    else:
        print(f"  新建extractor比重用快 {1/speedup:.2f}x ({((1/speedup-1)*100):.1f}%)")
    
    print(f"\n注意：重用extractor可能导致输出不更新的问题！")
    
    return times_new, times_reuse

def benchmark_inference(num_runs=100):
    """标准的推理速度测试（使用官方推荐方式）"""
    print(f"开始测试推理速度，运行 {num_runs} 次...")
    
    # 初始化模型（只需要一次）
    print("初始化模型...")
    init_model()
    
    # 准备输入数据（避免重复生成）
    torch.manual_seed(0)
    input_data = torch.rand(1, 3, 448, 448, dtype=torch.float)
    print(f'input_data.shape:{input_data.shape},dtype:{input_data.dtype}')
    
    # 预热几次
    print("预热中...")
    for i in range(10):
        result = test_inference_new_extractor(input_data)
        if i == 0:
            # 验证输出形状
            if isinstance(result, tuple):
                print(f"输出形状: out0={result[0].shape}, out1={result[1].shape}")
            else:
                print(f"输出形状: {result.shape}")
    
    print("开始正式测试...")
    # 正式测试
    times = []
    for i in range(num_runs):
        start_time = time.perf_counter_ns()
        result = test_inference_new_extractor(input_data)
        end_time = time.perf_counter_ns()
        
        inference_time_ms = (end_time - start_time) / 1_000_000  # 转换为毫秒
        times.append(inference_time_ms)
        
        if (i + 1) % 10 == 0:
            print(f"已完成 {i + 1}/{num_runs} 次推理，当前平均: {np.mean(times):.3f} ms")
    
    # 计算统计信息
    times = np.array(times)
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    median_time = np.median(times)
    
    print(f"\n=== 推理速度测试结果 ===")
    print(f"总运行次数: {num_runs}")
    print(f"平均推理时间: {avg_time:.3f} ms")
    print(f"中位数推理时间: {median_time:.3f} ms")
    print(f"最快推理时间: {min_time:.3f} ms")
    print(f"最慢推理时间: {max_time:.3f} ms")
    print(f"标准差: {std_time:.3f} ms")
    print(f"平均FPS: {1000/avg_time:.1f}")
    print(f"中位数FPS: {1000/median_time:.1f}")
    
    return avg_time, times

if __name__ == "__main__":
    # 运行对比测试
    benchmark_comparison(100)
    
    print("\n" + "="*50)
    
    # 运行异步测试
    test_async_inference(100)
    
    print("\n" + "="*50)
    
    # 运行标准测试
    benchmark_inference(100)
