# coding=utf-8
# Copyright 2023 Ling-Hao CHEN (https://lhchen.top) from Tsinghua University。
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, you need to cite as:
#
# @inproceedings{chen2023humanmac,
# 	title={HumanMAC: Masked Motion Completion for Human Motion Prediction},
# 	author={Chen, Ling-Hao and Zhang, Jiawei and Li, Yewen and Pang, Yiren and Xia, Xiaobo and Liu, Tongliang},
# 	journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
# 	year={2023}
# }
#
# If you use the code, you need to declare the copyright!
#
# Licensed under the MIT License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. We provide a license to use the code, 
# please read the specific details carefully.
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# visualization of the retargeted dataset

parents = [-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15]


def vis(x, y, z, name, iter=0):
    # rotation for visualization
    tmp = z
    z = y
    y = tmp

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = x + y + z
    ax.scatter(x, y, z, c=c, marker="*")
    for i in range(x.shape[0]):
        ax.text(x[i], y[i], z[i], str(i), fontsize=5)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    bone_len = []
    for i in range(1, x.shape[0]):
        xline = np.linspace(x[i], x[parents[i]], 100)
        yline = np.linspace(y[i], y[parents[i]], 100)
        zline = np.linspace(z[i], z[parents[i]], 100)
        bone_len.append(((x[i] - x[parents[i]])**2 + (y[i] -
                        y[parents[i]])**2 + (z[i] - z[parents[i]])**2)**(0.5))
        ax.plot(xline, yline, zline, color="gray")
    bone_len_str = ' '.join(["%.6f" % i for i in bone_len]) + '\n'

    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    plt.savefig(name+'/'+str(iter)+'.png')
    plt.close()
    return bone_len_str


n = 'vis'
if not os.path.exists(n):
    os.mkdir(n)
fulldata = np.load('retargeted.npy', allow_pickle=True)
print(fulldata.shape)
data = np.squeeze(fulldata[66], axis=1)
print(data.shape)
num_frames = data.shape[0]

for i in range(num_frames):
    frame = data[i].T
    vis(frame[0], frame[1], frame[2], n, i)
os.system('ffmpeg -r 60 -f image2 -i ./' + n + '/%d.png ' + n + '.mp4')
