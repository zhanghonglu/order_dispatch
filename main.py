import numpy as np
import pandas as pd

from collections import deque
import matplotlib.pyplot as plt
import argparse
import os.path
import torch
from model import QNetwork




if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    qnetwork_local = QNetwork(state_size=3, action_size=7, seed=0,fc1_uints=64, fc2_uints=64).to(device)

    # 模型参数文件
    model_file_name = "res\checkpoint_10_01.pth"
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"res\checkpoint_10_01.pth")

    #加载模型文件
    with torch.no_grad():
         qnetwork_local.load_state_dict(torch.load(file_path))
    qnetwork_local.eval()

    # 示例输入为bath_size *  state_action（10维）
    #state_action = [状态时间，状态经度，状态纬度，订单上车时间，订单下车时间，上车经度，上车纬度，下车经度，下车纬度，订单reward]
    value_test = qnetwork_local(torch.tensor([1.9219000e+04 ,1.0409080e+02, 3.0702640e+01, 1.9329000e+04 ,1.9801000e+04
, 1.0409652e+02 ,3.0692210e+01 ,1.0411526e+02, 3.0649590e+01, 2.5300000e+00]).unsqueeze(0).cuda())
    print(value_test.cpu().data.numpy())
    pass




