import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
from ultralytics import RTDETR
from thop import profile
import torch

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = r'D:\yolo11_wdq\ultralytics-8.3.90\runs_final\seed=0\MFFPN+GSConvE\train\weights\best.pt'
    model = YOLO(model_path) # 选择训练好的权重路径
    # model = RTDETR(model_path)
    result = model.val(
                       data=r'D:\yolo11_wdq\ultralytics-8.3.90\dataset_2\mydata.yaml',
                       # data=r'D:\yolo11_wdq\ultralytics-8.3.90\dataset_paper\mydata.yaml',
                        split='test', # split可以选择train、val、tes
                       # t 根据自己的数据集情况来选择.
                        imgsz=640,
                        batch=32,
                        # iou=0.7,
                        # rect=False,
                        # save_json=True, # ifyou need to cal coco metrice
                        project=r'D:\yolo11_wdq\ultralytics-8.3.90\runs_final\seed=0\MFFPN+GSConvE',
                        name='new',
                        # rect=False
                        )

    if model.task in ['detect', 'segment']:
        length = result.box.p.size
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        n_l, n_p, n_g, flops = model_info(model.model)

        # ====================== THOP GFLOPs (独立计算, 不影响原逻辑) ======================
        device = next(model.model.parameters()).device
        dummy_input = torch.randn(1, 3, 640, 640).to(device)

        try:
            thop_flops, thop_params = profile(
                model.model,
                inputs=(dummy_input,),
                verbose=False
            )
            thop_gflops = thop_flops * 2 / 1e9
        except Exception as e:
            print(f"[THOP] GFLOPs 计算失败: {e}")
            thop_gflops = None
        # ================================================================================

        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}',
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s',
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}',
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx in range(length):
            model_metrice_table.add_row([
                                        model_names[idx],
                                        f"{result.box.p[idx]:.4f}",
                                        f"{result.box.r[idx]:.4f}",
                                        f"{result.box.f1[idx]:.4f}",
                                        f"{result.box.ap50[idx]:.4f}",
                                        f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95
                                        f"{result.box.ap[idx]:.4f}"
                                    ])
        model_metrice_table.add_row([
                                    "all(平均数据)",
                                    f"{result.results_dict['metrics/precision(B)']:.4f}",
                                    f"{result.results_dict['metrics/recall(B)']:.4f}",
                                    f"{np.mean(result.box.f1[:length]):.4f}",
                                    f"{result.results_dict['metrics/mAP50(B)']:.4f}",
                                    f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
                                ])
        print(model_metrice_table)

        ####################################################
        print('\n' + '=' * 30)
        if thop_gflops is not None:
            print(f'[THOP] Model GFLOPs (640x640): {thop_gflops:.3f}')
        else:
            print('[THOP] Model GFLOPs: Failed')
        print('=' * 30)
        #####################################################

        with open(result.save_dir / 'paper_data.txt', 'w+', errors="ignore", encoding="utf-8") as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))
            #####################################################
            f.write('\n' + '=' * 30 + '\n')
            if thop_gflops is not None:
                f.write(f'[THOP] Model GFLOPs (640x640): {thop_gflops:.3f}\n')
            else:
                f.write('[THOP] Model GFLOPs: Failed\n')
            f.write('=' * 30 + '\n')
            #####################################################
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
#
#
# import warnings
# warnings.filterwarnings('ignore')
# import os
# import numpy as np
# from prettytable import PrettyTable
# from ultralytics import YOLO
# from ultralytics.utils.torch_utils import model_info
# from ultralytics import RTDETR
# from thop import profile
# import torch
#
# def get_weight_size(path):
#     stats = os.stat(path)
#     return f'{stats.st_size / 1024 / 1024:.1f}'
#
# if __name__ == '__main__':
#     model_path = r'D:\wdq\ultralytics-8.3.90\runs_posun\seed=0\ours-s\train\weights\best.pt'
#     model = YOLO(model_path) # 选择训练好的权重路径
#     # model = RTDETR(model_path)
#     result = model.val(data=r'D:\wdq\ultralytics-8.3.90\dataset_posun\mydata.yaml',
#                         split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
#                         imgsz=512,
#                         batch=32,
#                         # iou=0.7,
#                         # rect=False,
#                         # save_json=True, # if you need to cal coco metrice
#                         project=r'D:\wdq\ultralytics-8.3.90\runs_posun\seed=0\ours-s',
#                         name='dunhuang',
#                         # rect=False
#                         )
#
#     if model.task in ['detect', 'segment']:
#         length = result.box.p.size
#         model_names = list(result.names.values())
#         preprocess_time_per_image = result.speed['preprocess']
#         inference_time_per_image = result.speed['inference']
#         postprocess_time_per_image = result.speed['postprocess']
#         all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
#
#         n_l, n_p, n_g, flops = model_info(model.model)
#
#         # ====================== THOP GFLOPs (独立计算, 不影响原逻辑) ======================
#         device = next(model.model.parameters()).device
#         dummy_input = torch.randn(1, 3, 640, 640).to(device)
#
#         try:
#             thop_flops, thop_params = profile(
#                 model.model,
#                 inputs=(dummy_input,),
#                 verbose=False
#             )
#             thop_gflops = thop_flops * 2 / 1e9
#         except Exception as e:
#             print(f"[THOP] GFLOPs 计算失败: {e}")
#             thop_gflops = None
#         # ================================================================================
#
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
#
#         model_info_table = PrettyTable()
#         model_info_table.title = "Model Info"
#         model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
#         model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}',
#                                   f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s',
#                                   f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}',
#                                   f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
#         print(model_info_table)
#
#         model_metrice_table = PrettyTable()
#         model_metrice_table.title = "Model Metrice"
#         model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
#         for idx in range(length):
#             model_metrice_table.add_row([
#                                         model_names[idx],
#                                         f"{result.box.p[idx]:.4f}",
#                                         f"{result.box.r[idx]:.4f}",
#                                         f"{result.box.f1[idx]:.4f}",
#                                         f"{result.box.ap50[idx]:.4f}",
#                                         f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95
#                                         f"{result.box.ap[idx]:.4f}"
#                                     ])
#         model_metrice_table.add_row([
#                                     "all(平均数据)",
#                                     f"{result.results_dict['metrics/precision(B)']:.4f}",
#                                     f"{result.results_dict['metrics/recall(B)']:.4f}",
#                                     f"{np.mean(result.box.f1[:length]):.4f}",
#                                     f"{result.results_dict['metrics/mAP50(B)']:.4f}",
#                                     f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95
#                                     f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
#                                 ])
#         print(model_metrice_table)
#
#         ####################################################
#         print('\n' + '=' * 30)
#         if thop_gflops is not None:
#             print(f'[THOP] Model GFLOPs (640x640): {thop_gflops:.3f}')
#         else:
#             print('[THOP] Model GFLOPs: Failed')
#         print('=' * 30)
#         #####################################################
#
#         with open(result.save_dir / 'paper_data.txt', 'w+', errors="ignore", encoding="utf-8") as f:
#             f.write(str(model_info_table))
#             f.write('\n')
#             f.write(str(model_metrice_table))
#             #####################################################
#             f.write('\n' + '=' * 30 + '\n')
#             if thop_gflops is not None:
#                 f.write(f'[THOP] Model GFLOPs (640x640): {thop_gflops:.3f}\n')
#             else:
#                 f.write('[THOP] Model GFLOPs: Failed\n')
#             f.write('=' * 30 + '\n')
#             #####################################################
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)


# import warnings
# warnings.filterwarnings('ignore')
#
# import os
# import numpy as np
# from prettytable import PrettyTable
# from ultralytics import YOLO
# from ultralytics.utils.torch_utils import model_info
# from thop import profile
# import torch
#
#
# def get_weight_size(path):
#     stats = os.stat(path)
#     return f'{stats.st_size / 1024 / 1024:.1f}'
#
#
# if __name__ == '__main__':
#     model_path = r'D:\yolo11_wdq\ultralytics-8.3.90\runs_posun\seed=0\ours-n-SNI\train\weights\best.pt'
#     data_path = r'D:\yolo11_wdq\ultralytics-8.3.90\dataset_posun\mydata.yaml'
#
#     imgsz = 512
#     batch = 32
#
#     # ================== 加载模型 ==================
#     model = YOLO(model_path)
#
#     # ================== 验证 ==================
#     result = model.val(
#         data=data_path,
#         split='val',
#         imgsz=imgsz,
#         batch=batch,
#         project=r'D:\yolo11_wdq\ultralytics-8.3.90\runs_posun\seed=0\ours-n-SNI',
#         name='dunhuang'
#     )
#
#     # ================== 基本信息 ==================
#     model_names = list(result.names.values())
#
#     preprocess_time = result.speed['preprocess']
#     inference_time = result.speed['inference']
#     postprocess_time = result.speed['postprocess']
#     total_time = preprocess_time + inference_time + postprocess_time
#
#     n_l, n_p, n_g, flops = model_info(model.model)
#
#     # ================== THOP GFLOPs ==================
#     device = next(model.model.parameters()).device
#     dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
#
#     try:
#         thop_flops, thop_params = profile(
#             model.model,
#             inputs=(dummy_input,),
#             verbose=False
#         )
#         thop_gflops = thop_flops * 2 / 1e9
#     except Exception as e:
#         print(f"[THOP] GFLOPs 计算失败: {e}")
#         thop_gflops = None
#
#     print('-' * 20 + '论文数据如下' + '-' * 20)
#
#     # ================== Model Info 表 ==================
#     model_info_table = PrettyTable()
#     model_info_table.title = "Model Info"
#     model_info_table.field_names = [
#         "GFLOPs", "Parameters",
#         "Preprocess", "Inference", "Postprocess",
#         "FPS(All)", "FPS(Infer)",
#         "Model Size"
#     ]
#
#     model_info_table.add_row([
#         f'{flops:.1f}',
#         f'{n_p:,}',
#         f'{preprocess_time / 1000:.6f}s',
#         f'{inference_time / 1000:.6f}s',
#         f'{postprocess_time / 1000:.6f}s',
#         f'{1000 / total_time:.2f}',
#         f'{1000 / inference_time:.2f}',
#         f'{get_weight_size(model_path)}MB'
#     ])
#
#     print(model_info_table)
#
#     # ================== Metrics ==================
#     model_metrice_table = PrettyTable()
#
#     # ---------- Detection ----------
#     if model.task == 'detect':
#         length = result.box.p.size
#
#         model_metrice_table.title = "Detection Metrics"
#         model_metrice_table.field_names = [
#             "Class", "Precision", "Recall", "F1",
#             "mAP50", "mAP75", "mAP50-95"
#         ]
#
#         for i in range(length):
#             model_metrice_table.add_row([
#                 model_names[i],
#                 f"{result.box.p[i]:.4f}",
#                 f"{result.box.r[i]:.4f}",
#                 f"{result.box.f1[i]:.4f}",
#                 f"{result.box.ap50[i]:.4f}",
#                 f"{result.box.all_ap[i, 5]:.4f}",
#                 f"{result.box.ap[i]:.4f}"
#             ])
#
#         model_metrice_table.add_row([
#             "all",
#             f"{result.results_dict['metrics/precision(B)']:.4f}",
#             f"{result.results_dict['metrics/recall(B)']:.4f}",
#             f"{np.mean(result.box.f1):.4f}",
#             f"{result.results_dict['metrics/mAP50(B)']:.4f}",
#             f"{np.mean(result.box.all_ap[:, 5]):.4f}",
#             f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
#         ])
#
#     # ---------- Segmentation ----------
#     elif model.task == 'segment':
#         length = result.box.ap.size
#
#         model_metrice_table.title = "Segmentation Metrics"
#         model_metrice_table.field_names = [
#             "Class",
#             "Box mAP50", "Mask mAP50",
#             "Box mAP50-95", "Mask mAP50-95"
#         ]
#
#         for i in range(length):
#             model_metrice_table.add_row([
#                 model_names[i],
#                 f"{result.box.ap50[i]:.4f}",
#                 f"{result.seg.ap50[i]:.4f}",
#                 f"{result.box.ap[i]:.4f}",
#                 f"{result.seg.ap[i]:.4f}"
#             ])
#
#         model_metrice_table.add_row([
#             "all",
#             f"{result.results_dict['metrics/mAP50(B)']:.4f}",
#             f"{result.results_dict['metrics/mAP50(M)']:.4f}",
#             f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
#             f"{result.results_dict['metrics/mAP50-95(M)']:.4f}"
#         ])
#
#     print(model_metrice_table)
#
#     # ================== THOP 输出 ==================
#     print('\n' + '=' * 30)
#     if thop_gflops is not None:
#         print(f'[THOP] GFLOPs ({imgsz}x{imgsz}): {thop_gflops:.3f}')
#     else:
#         print('[THOP] GFLOPs: Failed')
#     print('=' * 30)
#
#     # ================== 保存 ==================
#     save_path = result.save_dir / 'paper_data.txt'
#
#     with open(save_path, 'w+', encoding="utf-8") as f:
#         f.write(str(model_info_table) + '\n')
#         f.write(str(model_metrice_table) + '\n')
#
#         f.write('\n' + '=' * 30 + '\n')
#         if thop_gflops is not None:
#             f.write(f'[THOP] GFLOPs ({imgsz}x{imgsz}): {thop_gflops:.3f}\n')
#         else:
#             f.write('[THOP] GFLOPs: Failed\n')
#         f.write('=' * 30 + '\n')
#
#     print('-' * 20, f'结果已保存至 {save_path}', '-' * 20)