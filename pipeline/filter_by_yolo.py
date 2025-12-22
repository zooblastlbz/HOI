"""
基于 YOLO 的数据筛选模块
从 LMDB 数据库或 JSON/JSONL 文件中读取数据，使用 YOLO 检测人物数量，
筛选出包含 1-5 人且每个人置信度 > 90% 的数据

使用方式（MPI 分布式）:
   mpirun --hostfile /etc/mpi/hostfile python filter_by_yolo.py --input_lmdb ... --output_lmdb ...
   mpirun --hostfile /etc/mpi/hostfile python filter_by_yolo.py --input_json ... --output_json ...
   mpirun --hostfile /etc/mpi/hostfile python filter_by_yolo.py --input_jsonl ... --output_jsonl ...
"""

import os
import json
import math
import pickle
import argparse
import tempfile
import glob
from typing import Optional, List, Tuple, Dict, Any

try:
    import lmdb
except ImportError:
    lmdb = None
    print("Warning: lmdb module not installed. Please install with: pip install lmdb")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics not installed. Please install with: pip install ultralytics")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import numpy as np


def load_json_records(path: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load JSON file, returning item list and wrapper info if present."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data, None
    
    if isinstance(data, dict):
        for key in ('results', 'data', 'items'):
            if key in data and isinstance(data[key], list):
                meta = {k: v for k, v in data.items() if k != key}
                return data[key], {'key': key, 'meta': meta}
    
    raise ValueError("JSON file must be a list or contain a list under 'results', 'data', or 'items'.")


def wrap_json_records(records: List[Dict[str, Any]], wrapper: Optional[Dict[str, Any]]) -> Any:
    """Reconstruct JSON structure if original file wrapped the list."""
    if wrapper is None:
        return records
    
    result = dict(wrapper.get('meta', {}))
    result[wrapper['key']] = records
    return result


def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse line in {path}: {e}")
    return records


def write_jsonl_records(records: List[Dict[str, Any]], path: str):
    """Write list of dicts to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_mpi_info():
    """通过环境变量获取 MPI rank 和 size"""
    # 获取当前进程的 rank
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', -1))
    if rank == -1:
        rank = int(os.environ.get('PMI_RANK', -1))  # 对于 MPICH 或 Intel MPI
    if rank == -1:
        rank = 0  # 非 MPI 环境默认为 0

    # 获取总进程数 size
    size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', -1))
    if size == -1:
        size = int(os.environ.get('PMI_SIZE', -1))  # 对于 MPICH 或 Intel MPI
    if size == -1:
        size = 1  # 非 MPI 环境默认为 1

    return rank, size



# MPI 信息
RANK, WORLD_SIZE = get_mpi_info()
LOCAL_RANK = RANK % 8 if RANK != -1 else 0

class YOLOFilter:
    """YOLO 筛选器"""
    
    def __init__(self,
                 model_path: str = "yolo11n.pt",
                 gpu_id: int = 0,
                 min_persons: int = 1,
                 max_persons: int = 5,
                 min_confidence: float = 0.9,
                 batch_size: int = 32):
        self.model_path = model_path
        self.device = f"cuda:{gpu_id}"
        self.min_persons = min_persons
        self.max_persons = max_persons
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.model = None
        
    def load_model(self):
        """加载 YOLO 模型"""
        if self.model is None:
            if YOLO is None:
                raise ImportError("ultralytics is not installed.")
            print(f"Rank {RANK}: Loading YOLO model on {self.device}...")
            self.model = YOLO(self.model_path)
            print(f"Rank {RANK}: YOLO model loaded")
    
    def get_image_path_from_data(self, data: dict) -> Optional[str]:
        """从 LMDB 数据中提取图片路径"""
        f_list = data.get('frame_dir_list', [])
        if isinstance(f_list, str):
            f_list = [f_list]
        
        if not f_list:
            return None
        
        raw_path = f_list[0]
        
        if os.path.isfile(raw_path):
            return raw_path
        else:
            potential_path = os.path.join(raw_path, '000001.jpg')
            if os.path.exists(potential_path):
                return potential_path
            return raw_path
    
    def process_batch(self, image_paths: List[str]) -> List[Tuple[bool, int, List[float]]]:
        """批量处理图片"""
        if self.model is None:
            self.load_model()
        
        # 过滤存在的图片
        valid_paths = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            if path and os.path.exists(path):
                valid_paths.append(path)
                valid_indices.append(i)
        
        # 初始化结果
        batch_results = [(False, 0, [])] * len(image_paths)
        
        if not valid_paths:
            return batch_results
        
        try:
            results = self.model(valid_paths, device=self.device, verbose=False)
            
            for idx, result in zip(valid_indices, results):
                if result.boxes is None:
                    continue
                
                boxes = result.boxes
                
                # 使用类别名称判断
                names = [result.names[cls.item()] for cls in boxes.cls.int()]
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                
                # 筛选 person 类别
                person_confidences = [conf for name, conf in zip(names, confs) if name == 'person']
                person_count = len(person_confidences)
                
                # 检查条件
                if person_count < self.min_persons or person_count > self.max_persons:
                    batch_results[idx] = (False, person_count, person_confidences)
                elif person_count > 0 and all(conf >= self.min_confidence for conf in person_confidences):
                    batch_results[idx] = (True, person_count, person_confidences)
                else:
                    batch_results[idx] = (False, person_count, person_confidences)
                    
        except Exception as e:
            print(f"Rank {RANK}: Error in batch processing: {e}")
        
        return batch_results

    def filter_json_records(self, records: List[Dict[str, Any]], image_key: str = "image_path") -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """筛选 JSON/JSONL 数据，返回符合条件的记录列表和统计信息"""
        stats = {
            'total_processed': 0,
            'valid_count': 0,
            'invalid_count': 0,
        }
        filtered_records: List[Dict[str, Any]] = []
        batch_records: List[Dict[str, Any]] = []
        batch_paths: List[Optional[str]] = []

        def flush_batch():
            if not batch_records:
                return
            results = self.process_batch(batch_paths)
            for record, (is_valid, person_count, confidences) in zip(batch_records, results):
                stats['total_processed'] += 1
                if is_valid:
                    record_copy = dict(record)
                    safe_confidences = [float(c) for c in confidences]
                    record_copy['yolo_filter'] = {
                        'person_count': person_count,
                        'person_confidences': safe_confidences,
                    }
                    filtered_records.append(record_copy)
                    stats['valid_count'] += 1
                else:
                    stats['invalid_count'] += 1
            batch_records.clear()
            batch_paths.clear()

        for record in records:
            batch_records.append(record)
            batch_paths.append(record.get(image_key))
            if len(batch_records) >= self.batch_size:
                flush_batch()

        flush_batch()
        return filtered_records, stats
    
    def filter_lmdb(self,
                    input_lmdb_path: str,
                    output_lmdb_path: str,
                    start_idx: int,
                    end_idx: int,
                    map_size: int = 1099511627776):
        """筛选 LMDB 数据库的指定范围"""
        if lmdb is None:
            raise ImportError("lmdb module not installed.")
        
        self.load_model()
        
        print(f"Rank {RANK}: Processing [{start_idx}, {end_idx})")
        
        env_in = lmdb.open(input_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        output_dir = os.path.dirname(output_lmdb_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        env_out = lmdb.open(output_lmdb_path, map_size=map_size)
        
        stats = {
            'total_processed': 0,
            'valid_count': 0,
            'invalid_count': 0,
        }
        
        with env_in.begin() as txn_in:
            output_idx = 0
            batch_keys = []
            batch_data = []
            batch_paths = []
            
            with env_out.begin(write=True) as txn_out:
                for i in tqdm(range(start_idx, end_idx), desc=f"Rank {RANK}", disable=(RANK != 0)):
                    key = f'{i}'.encode()
                    value_bytes = txn_in.get(key)
                    
                    if value_bytes is None:
                        continue
                    
                    try:
                        data = pickle.loads(value_bytes)
                    except Exception:
                        continue
                    
                    image_path = self.get_image_path_from_data(data)
                    batch_keys.append(i)
                    batch_data.append(data)
                    batch_paths.append(image_path)
                    
                    # 批量处理
                    if len(batch_keys) >= self.batch_size:
                        results = self.process_batch(batch_paths)
                        
                        for j, (is_valid, person_count, confidences) in enumerate(results):
                            stats['total_processed'] += 1
                            
                            if is_valid:
                                batch_data[j]['yolo_filter'] = {
                                    'person_count': person_count,
                                    'person_confidences': confidences,
                                }
                                out_key = f'{output_idx}'.encode()
                                txn_out.put(out_key, pickle.dumps(batch_data[j]))
                                output_idx += 1
                                stats['valid_count'] += 1
                            else:
                                stats['invalid_count'] += 1
                        
                        batch_keys = []
                        batch_data = []
                        batch_paths = []
                
                # 处理剩余
                if batch_keys:
                    results = self.process_batch(batch_paths)
                    
                    for j, (is_valid, person_count, confidences) in enumerate(results):
                        stats['total_processed'] += 1
                        
                        if is_valid:
                            batch_data[j]['yolo_filter'] = {
                                'person_count': person_count,
                                'person_confidences': confidences,
                            }
                            out_key = f'{output_idx}'.encode()
                            txn_out.put(out_key, pickle.dumps(batch_data[j]))
                            output_idx += 1
                            stats['valid_count'] += 1
                        else:
                            stats['invalid_count'] += 1
        
        env_in.close()
        env_out.close()
        
        print(f"Rank {RANK}: Processed {stats['total_processed']}, Valid {stats['valid_count']}")
        
        return stats


def merge_lmdb_databases(lmdb_paths: List[str], output_path: str, map_size: int = 1099511627776):
    """合并多个 LMDB 数据库"""
    print(f"Merging {len(lmdb_paths)} LMDB databases...")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    env_out = lmdb.open(output_path, map_size=map_size)
    output_idx = 0
    
    with env_out.begin(write=True) as txn_out:
        for lmdb_path in lmdb_paths:
            if not os.path.exists(lmdb_path):
                continue
            
            env_in = lmdb.open(lmdb_path, readonly=True, lock=False)
            
            with env_in.begin() as txn_in:
                cursor = txn_in.cursor()
                for key, value in cursor:
                    out_key = f'{output_idx}'.encode()
                    txn_out.put(out_key, value)
                    output_idx += 1
            
            env_in.close()
    
    env_out.close()
    print(f"Merged {output_idx} samples into {output_path}")




def main():
    parser = argparse.ArgumentParser(description="Filter data using YOLO (MPI)")
    parser.add_argument("--input_lmdb", type=str, default=None)
    parser.add_argument("--output_lmdb", type=str, default=None)
    parser.add_argument("--input_json", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223.json", help="Input JSON file (list or dict with results)")
    parser.add_argument("--output_json", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223_yolo_filter.json", help="Output JSON file for filtered data")
    parser.add_argument("--input_jsonl", type=str, default=None, help="Input JSONL file (one JSON object per line)")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Output JSONL file for filtered data")
    parser.add_argument("--image_key", type=str, default="image_path", help="JSON key for image path")
    parser.add_argument("--model", type=str, default="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/yolo12x.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_persons", type=int, default=1)
    parser.add_argument("--max_persons", type=int, default=5)
    parser.add_argument("--min_confidence", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--map_size", type=int, default=1099511627776)
    
    parser.add_argument("--temp_dir", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223_tmp", help="Temporary directory for intermediate results")
    
    
    args = parser.parse_args()
    
    if RANK == 0:
        print(f"MPI: {WORLD_SIZE} processes")
    
    # 每个进程使用本地 GPU
    local_gpu_id = LOCAL_RANK
    temp_dir = args.temp_dir if args.temp_dir else tempfile.gettempdir()
    
    yolo_filter = YOLOFilter(
        model_path=args.model,
        gpu_id=local_gpu_id,
        min_persons=args.min_persons,
        max_persons=args.max_persons,
        min_confidence=args.min_confidence,
        batch_size=args.batch_size
    )

    # JSON / JSONL 模式
    if args.input_json or args.input_jsonl:
        input_mode = "json" if args.input_json else "jsonl"
        input_path = args.input_json if args.input_json else args.input_jsonl
        
        if input_mode == "json":
            default_output = os.path.splitext(input_path)[0] + "_yolo_filtered.json"
            output_path = args.output_json if args.output_json else default_output
        else:
            base, ext = os.path.splitext(input_path)
            if ext.lower() not in (".jsonl", ".jsonlines"):
                ext = ".jsonl"
            default_output = base + "_yolo_filtered" + ext
            output_path = args.output_jsonl if args.output_jsonl else default_output

        if RANK == 0:
            print(f"Input mode: {input_mode.upper()}, Input: {input_path}")
            print(f"Output path: {output_path}")
            if not os.path.exists(input_path):
                print(f"Error: Input file {input_path} not found.")
                return
        
        try:
            if input_mode == "json":
                records, wrapper_info = load_json_records(input_path)
            else:
                records = load_jsonl_records(input_path)
                wrapper_info = None
        except Exception as e:
            if RANK == 0:
                print(f"Failed to load input file: {e}")
            return
        
        if args.num_samples:
            records = records[:args.num_samples]
        
        total_entries = len(records)
        samples_per_process = math.ceil(total_entries / WORLD_SIZE) if WORLD_SIZE > 0 else total_entries
        start_idx = RANK * samples_per_process
        end_idx = min(start_idx + samples_per_process, total_entries)

        print(f"Rank {RANK}: Processing {len(records[start_idx:end_idx])} samples (index {start_idx} to {end_idx})")

        # 临时输出路径
        os.makedirs(temp_dir, exist_ok=True)
        marker_dir = os.path.join(temp_dir, "yolo_filter_markers")
        os.makedirs(marker_dir, exist_ok=True)
        temp_output = os.path.join(temp_dir, f"yolo_filter_rank{RANK}_{os.getpid()}")
        temp_output = temp_output + (".json" if input_mode == "json" else ".jsonl")

        marker_file = os.path.join(marker_dir, f"rank_{RANK}.txt")
        with open(marker_file, 'w') as f:
            f.write(temp_output)

        # 筛选并写出当前进程结果
        filtered_records, stats = yolo_filter.filter_json_records(records[start_idx:end_idx], image_key=args.image_key)
        if input_mode == "json":
            with open(temp_output, 'w', encoding='utf-8') as f:
                json.dump(filtered_records, f, ensure_ascii=False, indent=2)
        else:
            write_jsonl_records(filtered_records, temp_output)

        done_file = os.path.join(marker_dir, f"rank_{RANK}.done")
        with open(done_file, 'w') as f:
            f.write("done")

        if RANK == 0:
            import time
            import shutil

            if total_entries:
                print(f"Total entries: {total_entries}, Per process: ~{samples_per_process}")
            else:
                print("No entries found in input.")
            
            # 等待所有进程完成
            print("Waiting for all processes to complete...")
            while True:
                done_files = glob.glob(os.path.join(marker_dir, "*.done"))
                if len(done_files) >= WORLD_SIZE:
                    break
                time.sleep(1)

            # 读取所有临时路径
            temp_paths = []
            for r in range(WORLD_SIZE):
                m_file = os.path.join(marker_dir, f"rank_{r}.txt")
                if os.path.exists(m_file):
                    with open(m_file, 'r') as f:
                        temp_paths.append(f.read().strip())

            # 合并
            merged_records: List[Dict[str, Any]] = []
            for path in temp_paths:
                if not os.path.exists(path):
                    continue
                try:
                    if input_mode == "json":
                        with open(path, 'r', encoding='utf-8') as f:
                            merged_records.extend(json.load(f))
                    else:
                        merged_records.extend(load_jsonl_records(path))
                except Exception as e:
                    print(f"Warning: failed to merge {path}: {e}")
            
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            if input_mode == "json":
                output_obj = wrap_json_records(merged_records, wrapper_info)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_obj, f, ensure_ascii=False, indent=2)
            else:
                write_jsonl_records(merged_records, output_path)

            # 清理临时文件
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)
            shutil.rmtree(marker_dir, ignore_errors=True)

            print(f"Output saved to: {output_path}")
            print(f"Rank {RANK}: Processed {stats['total_processed']}, Valid {stats['valid_count']}")
        else:
            # 非 0 号进程完成后直接退出
            print(f"Rank {RANK}: Processed {stats['total_processed']}, Valid {stats['valid_count']}")
        return

    # LMDB 模式
    # 获取总数据量
    env = lmdb.open(args.input_lmdb, readonly=True, lock=False)
    with env.begin() as txn:
        total_entries = txn.stat()['entries']
    env.close()
    
    if args.num_samples:
        total_entries = min(total_entries, args.num_samples)
    
    # 计算每个进程的数据范围
    samples_per_process = total_entries // WORLD_SIZE
    start_idx = RANK * samples_per_process
    end_idx = start_idx + samples_per_process if RANK < WORLD_SIZE - 1 else total_entries
    
    if RANK == 0:
        print(f"Total: {total_entries}, Per process: ~{samples_per_process}")
    
    # 临时输出路径
    os.makedirs(temp_dir, exist_ok=True)
    temp_output = os.path.join(temp_dir, f"yolo_filter_rank{RANK}_{os.getpid()}")
    
    # 写入标记文件，供 rank 0 发现
    marker_dir = os.path.join(temp_dir, "yolo_filter_markers")
    os.makedirs(marker_dir, exist_ok=True)
    marker_file = os.path.join(marker_dir, f"rank_{RANK}.txt")
    with open(marker_file, 'w') as f:
        f.write(temp_output)
    
    # 筛选
    yolo_filter.filter_lmdb(
        input_lmdb_path=args.input_lmdb,
        output_lmdb_path=temp_output,
        start_idx=start_idx,
        end_idx=end_idx,
        map_size=args.map_size // WORLD_SIZE
    )
    
    # 写入完成标记
    done_file = os.path.join(marker_dir, f"rank_{RANK}.done")
    with open(done_file, 'w') as f:
        f.write("done")
    
    # Rank 0 等待所有进程完成并合并结果
    if RANK == 0:
        import time
        
        # 等待所有进程完成
        print("Waiting for all processes to complete...")
        while True:
            done_files = glob.glob(os.path.join(marker_dir, "*.done"))
            if len(done_files) >= WORLD_SIZE:
                break
            time.sleep(1)
        
        # 读取所有临时路径
        temp_paths = []
        for r in range(WORLD_SIZE):
            marker_file = os.path.join(marker_dir, f"rank_{r}.txt")
            with open(marker_file, 'r') as f:
                temp_paths.append(f.read().strip())
        
        # 合并
        merge_lmdb_databases(temp_paths, args.output_lmdb, args.map_size)
        
        # 清理临时文件
        import shutil
        for path in temp_paths:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(marker_dir, ignore_errors=True)
        
        print(f"Output saved to: {args.output_lmdb}")


if __name__ == "__main__":
    main()
