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

# Define the parent-child relationships between points in a skeleton for the AMASS dataset.
parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
           8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Define a function for visualizing 3D pose data.


def vis(x, y, z, name, iter=0):
    # Rotate the data for visualization (not used in this code).
    tmp = z
    z = y
    y = tmp

    # Create a 3D scatter plot of the data points.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = x + y + z
    ax.scatter(x, y, z, c=c, marker="*")
    for i in range(x.shape[0]):
        ax.text(x[i], y[i], z[i], str(i), fontsize=5)

    # Set the axis limits for the plot.
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

    # Set labels for the axes.
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    # Save the plot as an image with a name based on 'name' and 'iter'.
    plt.savefig(name+'/'+str(iter)+'.png')
    plt.close()
    return bone_len_str


# Create a directory 'vis' if it doesn't exist.
n = 'vis'
if not os.path.exists(n):
    os.mkdir(n)
fulldata = np.load('amass.npy', allow_pickle=True)
print(fulldata.shape)

# Extract and preprocess data for a specific example (sample 66).
data = np.squeeze(fulldata[66], axis=1)
print(data.shape)
num_frames = data.shape[0]

# Add a root position (all zeros) to the data.
root = np.zeros((num_frames, 1, 3))
data = np.concatenate((root, data), axis=1)

# Iterate through the frames and call the 'vis' function for each frame.
for i in range(num_frames):
    frame = data[i].T
    print(frame.shape)
    vis(frame[0], frame[1], frame[2], n, i)

# Use FFmpeg to combine the images into a video named 'vis.mp4'.
os.system('ffmpeg -r 60 -f image2 -i ./' + n + '/%d.png ' + n + '.mp4')
